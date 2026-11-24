"""AOrchestra baseline adapted from the authors' implementation (Apache-2.0); see THIRD_PARTY_NOTICES.md."""
from __future__ import annotations

import json

from rci.agents.tooling import STR, AgentCtx, ToolResult, ToolSpec
from rci.protocols.base import Protocol

COORD_PROMPT = """You are the MainAgent (Orchestrator). Your task is to solve the given TASK by decomposing it into subtasks and delegating each to a sub-agent. You never use the task tools yourself; you have EXACTLY TWO tools: `delegate_task` and `{submit}`. Call exactly one tool per turn.

DECISION PROCESS:
1. REVIEW the SUBTASK HISTORY below - check status, result, and key findings of each attempt
2. EVALUATE: Do the results SUFFICIENTLY answer the TASK?
   - If any subtask returned a valid result with status "done" → Consider using '{submit}'
   - If subtask status is "partial" or "timeout" → Review its key findings to see what was accomplished
3. DECIDE next action:
   - Results sufficient → Use '{submit}' with the answer
   - Need more work → Use 'delegate_task' for the REMAINING work (don't repeat what's done)

BUDGET AWARENESS:
- You have LIMITED attempts (see Progress below)
- Each delegation costs time and resources
- If a result looks correct and was verified, trust it and submit

When delegating, write a SPECIFIC, ACTIONABLE task_instruction, pass the relevant findings from previous attempts as context, and select the relevant tools from TOOLS AVAILABLE TO SUB-AGENTS.

==== Progress ====
[Attempt {attempt}/{max}] Remaining {remaining} attempts
⚠️ Budget is limited. Make each attempt count.

==== TASK ====
{task}

==== SUBTASK HISTORY ====
{history}

==== TOOLS AVAILABLE TO SUB-AGENTS ====
{tools}"""

SUB_PROMPT = """You are a specialized SubAgent. Complete the assigned task efficiently.

==== Your Task (from MainAgent) ====
{instruction}

==== Context ====
{context}

==== Original Question (for reference) ====
{task}

==== Guidelines ====
1. Focus on completing YOUR TASK above
2. Think step by step before acting
3. Once done, use 'finish' IMMEDIATELY

⚠️ BUDGET: When remaining_steps <= 5, use 'finish' NOW!"""

TRACE_PROMPT = """You are a trajectory summarizer. Review the SubAgent's execution trace.

Task: {instruction}
Steps: {n}

=== Trace ===
{trace}
===

Summarize in 5-10 bullets: key progress, problems, remaining issues.
Output ONLY bullets."""


def sub_progress(step: int, max_steps: int) -> str:
    r = max_steps - step
    text = f"[Step {step}/{max_steps}] Remaining {r} steps"
    if r <= 3:
        text += f"\n🚨 CRITICAL: Only {r} steps left! Use 'finish' NOW!"
    elif r <= 5:
        text += f"\n⚠️ Warning: {r} steps remaining. Plan to finish soon."
    return text


class AOrchestraProtocol(Protocol):
    name = "aorchestra"

    def __init__(self, max_attempts: int = 10, sub_max_turns: int = 50, **kw):
        super().__init__(**kw)
        self.max_attempts, self.sub_max_turns = max_attempts, sub_max_turns

    def build(self, run) -> None:
        task = run.task
        self.task_tools_proto = task.tools(run.example, run)
        st = {"attempt": 0, "entries": [], "subs": {}, "pending_report": [], "final": None, "coord": 0,
              "calls_in_turn": {}}
        run.state["aor"] = st
        submit = task.submit_tool(run.example, with_refs=False)
        cfg = run.cfg.agent_config()
        cfg.stateless, cfg.summarize = True, False
        cfg.max_actions = 10 * self.max_attempts
        coord = run.add_agent(task.system_prompt(run.example),
                              "You are the MainAgent. The current state of the task is shown below.",
                              [self._delegate_tool(run), submit], failing=run.cfg.n_failing > 0, role="coordinator",
                              cfg=cfg)
        st["coord"] = coord.id

    def _tool_desc(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.task_tools_proto)

    def _delegate_tool(self, run) -> ToolSpec:
        names = [t.name for t in self.task_tools_proto]

        def delegate(ctx: AgentCtx, args: dict) -> ToolResult:
            st = run.state["aor"]
            instr = str(args.get("task_instruction", "")).strip()
            if not instr:
                return ToolResult("Error: task_instruction is required.", ok=False)
            sel = [n for n in (args.get("tools") or names) if n in names] or names
            sub = self._spawn(run, instr, str(args.get("context", "") or "None"), sel)
            coord = run.agents[st["coord"]]
            coord.wait(lambda: sub.id not in st["subs"] or st["subs"][sub.id]["reported"], until_step=10 ** 9)
            return ToolResult(f"Delegated to sub-agent {sub.id}; you will see its report in the SUBTASK HISTORY.")

        return ToolSpec("delegate_task", "Delegate a subtask to a new sub-agent with the selected tools.", {
            "type": "object", "properties": {
                "task_instruction": {**STR, "description": "Specific, actionable instruction for the sub-agent"},
                "context": {**STR, "description": "Relevant findings from previous attempts"},
                "tools": {"type": "array", "items": {"type": "string", "enum": names},
                          "description": "Task tools the sub-agent may use (default: all)"}},
            "required": ["task_instruction"]}, delegate)

    def _spawn(self, run, instr: str, context: str, tool_names: list[str]):
        task, st = run.task, run.state["aor"]
        props, req = task.answer_properties()
        holder = {}

        def finish(ctx: AgentCtx, args: dict) -> ToolResult:
            status = args.get("status") if args.get("status") in ("done", "partial", "blocked") else "partial"
            rep = {"result": args.get("result"), "status": status, "summary": str(args.get("summary", ""))}
            return ToolResult("Reported to MainAgent.", final=rep)

        fin = ToolSpec("finish", "Report your result to MainAgent. Use when you have found the answer or cannot "
                                 "proceed. When remaining steps < 5, finish with your best result.", {
            "type": "object", "properties": {
                "result": {"type": "object", "properties": props, "description": "your answer, in the task's answer schema"},
                "status": {"type": "string", "enum": ["done", "partial", "blocked"]},
                "summary": {**STR, "description": "brief summary of what you did"}},
            "required": ["result", "status", "summary"]}, finish, terminal=True)
        tools = [t for t in task.tools(run.example, run) if t.name in tool_names] + [fin]
        cfg = run.cfg.agent_config()
        cfg.max_turns = self.sub_max_turns
        system = SUB_PROMPT.format(instruction=instr, context=context, task=task.task_prompt(run.example, 0, 1, run.rng))
        system += "\n\n" + task.system_prompt(run.example)
        sub = run.add_agent(system, "Begin your assigned task.", tools, role="subagent", cfg=cfg)
        st["subs"][sub.id] = {"instruction": instr, "reported": False, "attempt": st["attempt"]}
        return sub

    def live_block(self, run, agent, step: int) -> str | None:
        st = run.state["aor"]
        if agent.id == st["coord"]:
            hist = self._history(st) or "No subtasks completed yet."
            return COORD_PROMPT.format(submit=agent.submit_name, attempt=st["attempt"] + 1, max=self.max_attempts,
                                       remaining=self.max_attempts - st["attempt"],
                                       task=run.task.task_prompt(run.example, 0, 1, run.rng), history=hist,
                                       tools=self._tool_desc())
        return sub_progress(len(agent.turns) + 1, self.sub_max_turns)

    def gate(self, run, agent, tool_name: str) -> str | None:
        st = run.state["aor"]
        if agent.id != st["coord"]:
            return None
        key = (agent.id, run.step)
        st["calls_in_turn"][key] = st["calls_in_turn"].get(key, 0) + 1
        if st["calls_in_turn"][key] > 1:
            return "Rejected: you must call exactly one tool per turn."
        return None

    def on_turn_end(self, run, agent, step: int) -> None:
        st = run.state["aor"]
        if agent.id != st["coord"]:
            return
        st["attempt"] += 1
        if agent.done:
            st["final"] = agent.final
        elif st["attempt"] >= self.max_attempts:
            agent.done = True

    def needs_final(self, run, agent) -> bool:
        return False

    def on_agent_done(self, run, agent) -> None:
        st = run.state["aor"]
        if agent.id in st["subs"] and not st["subs"][agent.id]["reported"]:
            st["pending_report"].append(agent.id)

    async def after_step(self, run, step: int) -> None:
        st = run.state["aor"]
        while st["pending_report"]:
            sid = st["pending_report"].pop(0)
            sub = run.agents[sid]
            rep = sub.final or {"result": None, "status": "timeout",
                                "summary": f"Used all {self.sub_max_turns} steps without finish"}
            trace = self._trace_text(sub)
            call = await run.llm_for(sid).chat(
                [{"role": "user", "content": TRACE_PROMPT.format(instruction=st["subs"][sid]["instruction"][:200],
                                                                 n=len(sub.turns), trace=trace)}],
                max_tokens=2048, temperature=run.cfg.temperature, enable_thinking=False, tag="trace_summary")
            sub.calls.append(call)
            st["entries"].append({"sub": sid, "attempt": st["subs"][sid]["attempt"], "steps": len(sub.turns),
                                  "task": st["subs"][sid]["instruction"], **rep,
                                  "trace_summary": (call.content or "").strip()})
            st["subs"][sid]["reported"] = True

    @staticmethod
    def _trace_text(sub) -> str:
        lines = []
        for k, entry in enumerate(e for e in sub.full_log if "messages" in e):
            res = {m.get("tool_call_id"): str(m.get("content", "")) for m in entry["messages"] if m["role"] == "tool"}
            for m in entry["messages"]:
                for tc in m.get("tool_calls", []) if m["role"] == "assistant" else []:
                    out = res.get(tc["id"], "")
                    lines.append(f"Step {k + 1}: {tc['function']['name']}({tc['function']['arguments'][:200]}) -> {out[:300]}")
        return "\n".join(lines) or "(no actions)"

    @staticmethod
    def _history(st) -> str:
        out = []
        for i, e in enumerate(st["entries"], 1):
            mark = "✅" if e["status"] == "done" else "⚠️"
            res = json.dumps(e["result"], ensure_ascii=False) if e.get("result") else "(no result)"
            out.append(f"[Attempt {i}] {mark} {e['status']} | Steps: {e['steps']}/50\n├─ Task: {e['task']}\n"
                       f"├─ Result: {res}\n├─ Summary: {e.get('summary', '')}\n└─ Trace summary:\n   {e['trace_summary']}")
        if out:
            done = sum(1 for e in st["entries"] if e["status"] == "done")
            out.append(f"---\nSummary: {done}/{len(st['entries'])} subtasks done")
        return "\n".join(out)

    def finished(self, run) -> bool:
        return run.agents[run.state["aor"]["coord"]].done

    def collect(self, run) -> list[dict]:
        st = run.state["aor"]
        coord = run.agents[st["coord"]]
        final = coord.final
        if final is None:
            if run.task.name == "audit":
                final = run.task.recorded_answer([e["result"] for e in st["entries"] if isinstance(e.get("result"), dict)])
            else:
                done = [e for e in st["entries"] if e["status"] == "done" and e.get("result")]
                part = [e for e in st["entries"] if e.get("result")]
                pick = (done or part or [None])[-1]
                final = pick["result"] if pick else None
        return [{"agent": coord.id, "failing": coord.failing, "role": "coordinator", "final": final,
                 "forced": coord.final is None}]

    def dump(self, run) -> dict:
        st = run.state["aor"]
        return {"attempts": st["attempt"], "entries": st["entries"], "n_subagents": len(st["subs"])}
