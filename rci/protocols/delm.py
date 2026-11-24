"""DeLM baseline adapted from the authors' implementation (MIT); see THIRD_PARTY_NOTICES.md."""
from __future__ import annotations

import re

from rci.agents.tooling import STR, AgentCtx, ToolResult, ToolSpec
from rci.agents.tokens import count_tokens, truncate_to_tokens
from rci.protocols.base import Protocol, is_failing

NOTE_TYPES = ("FACT", "TRIED", "OBSERVED", "FAIL")
MAX_NOTE_CHARS, MAX_RESULT_CHARS = 100, 300
CHARS_PER_TOKEN, FORMAT_OVERHEAD = 4, 25
_CONCRETE_TOKENS = ("::", "FAILED", "PASSED", "ERROR", "Exception", "Traceback", ".py", "/", "()")
_CONCRETE_PATTERNS = (re.compile(r"\b(?:exit|status|code)\s*[=:]?\s*\d+\b", re.I), re.compile(r"\bline\s+\d+\b", re.I),
                      re.compile(r"\b[\w./-]+\.py(?::\d+)?\b"))
_ROUTINE = ("search", "searched", "searching", "search_file", "grep", "grepped", "grepping", "find", "found",
            "finding", "ls", "list", "listed", "listing", "open", "opened", "opening", "view", "viewed", "viewing",
            "cat", "less", "head", "tail", "look", "looked", "looking", "ran ls", "ran grep", "ran find",
            "ran search", "ran search_file", "fetch", "fetched", "fetching")
_FAIL_HINTS = ("fail", "failed", "incorrect", "wrong", "invalid", "no match", "no matches", "not found", "no result",
               "no results", "empty", "could not", "couldn't", "doesn't exist", "no such", "0 result", "exception",
               "error", "broke", "broken")

DELM_COMMON = """==== PARALLEL AGENTS AND THE SHARED BOARD ====
You are agent a{i}, one of {N} agents solving this task in parallel. Agents do NOT see each other's context or workspace — only the SHARED BOARD, which is shown to you before each of your turns. Cooperate via the board. Every agent submits its own complete answer with `{submit}`.

The original task remains authoritative. Queue items divide work; their summaries never replace or weaken its requirements.

You may use the task tools only while you hold a claimed item. Claim an open, ready item with `claim_task`; first claim wins; one unfinished claim per agent. Finish it with `complete_task`, stating results, limitations and checks already obtained, plus your candidate answer if you have one.
Prefer complementary items that can begin from available inputs. Do not invent work or duplicate an existing item merely to occupy all agents.
When no open item is ready and required work remains, add it with `add_tasks`. If no required work remains, submit.
Wait (`wait_board`) only for a specific required input from a running peer. Never wait for an optional addition.

NOTE writes short typed entries to the SHARED LESSONS blackboard so other parallel agents can see your findings.
TYPE must be EXACTLY one of: FACT | TRIED | OBSERVED | FAIL
  - FACT: objective discovery ({fact_examples})
  - TRIED: action you just took (one-line summary)
  - OBSERVED: empirical result ({observed_examples})
  - FAIL: a failed attempt with brief reason (so other agents don't repeat it)
Publish only evidence — let other agents draw their own conclusions. Content size <=100 chars. Write 0-3 entries per turn.
Your `complete_task` summary is your RESULT: result=<one-sentence result> | evidence=<what verified it> | risk=<known weakness>, <=300 chars. Be specific and honest about evidence and risk.

Before submission, read the board and address corrections relevant to your final answer. Once evidence supports all requirements and no relevant failure remains, submit. Other agents still working do not block your submission."""

DELM_DECOMPOSER = """You are the first agent. Your first action must be `add_tasks`: inspect the task and publish independently actionable items with their dependencies, before completing a whole-task plan. Then claim one and work on it in this same session. Publish further independent work as its boundary becomes clear. If no split is yet useful, publish one item for the whole task. Do not manufacture {N} components."""

DELM_WORKER = """Leave initial item creation to the first agent. Until the queue exists, call `wait_board`. Then claim an open, ready item; respect the returned owner."""

GUIDANCE = """Read these before each action:
- [Results] are PEER-PROPOSED CANDIDATE results. Read them BEFORE committing to your own approach: if a peer's result covers your idea and the evidence holds up, corroborate it rather than re-deriving it; if the evidence is thin, propose your own. Do not blindly copy — verify before adopting.
- Avoid repeating the SAME mistake/reason named in peer FAIL entries, but do NOT abandon a correct target just because a peer's execution failed there.
- Build on FACT entries (objective findings). OBSERVED entries are empirical results; TRIED entries are actions taken."""

EXAMPLES = {"audit": ("snippet id and the exact text", "a quote that conflicts with another snippet"),
            "browsecomp": ("document id, entity, date", "a search that returned the candidate"),
            "fate": ("lemma name, Lean error text", "check_lean output"),
            "swe": ("file path, function signature, error text", "test output, exit code")}


def _is_concrete(c: str) -> bool:
    low = c.lower()
    return any(t.lower() in low for t in _CONCRETE_TOKENS) or any(p.search(c) for p in _CONCRETE_PATTERNS)


def _is_routine_tried(c: str) -> bool:
    if not c or _is_concrete(c):
        return False
    low = c.lower().strip()
    if any(h in low for h in _FAIL_HINTS):
        return False
    first_two = " ".join(low.split()[:2])
    first = first_two.split(" ")[0] if first_two else ""
    return any(first_two.startswith(v) or first == v for v in _ROUTINE)


def _priority(e: dict) -> int:
    t = e["type"]
    if t == "FACT" or (t == "OBSERVED" and _is_concrete(e["content"])):
        return 0
    return {"FAIL": 1, "OBSERVED": 2, "TRIED": 3}.get(t, 4)


class Board:
    def __init__(self, step_fn):
        self.items: list[dict] = []
        self.notes: list[dict] = []
        self.attachments: dict[str, str] = {}
        self.version = 0
        self.handoff_done = False
        self.step_fn = step_fn

    def bump(self):
        self.version += 1

    def _select(self, window_tokens: int = 2000, result_tokens: int = 200, max_results: int = 4) -> list[dict]:
        durable, results, index = [], [], {}
        for e in self.notes:
            if e["type"] == "RESULT":
                results.append(dict(e, _dup=1))
                continue
            key = (e["type"], " ".join(e["content"].split()))
            if key in index:
                index[key]["_dup"] += 1
                index[key]["_last"] = max(index[key]["_last"], e["step"])
            else:
                ee = dict(e, _dup=1, _last=e["step"])
                index[key] = ee
                durable.append(ee)
        size = lambda e: FORMAT_OVERHEAD + len(e["content"]) + (len(f" (x{e['_dup']})") if e["_dup"] > 1 else 0)
        kept, used, kept_ids = [], 0, set()
        budget = window_tokens * CHARS_PER_TOKEN
        reserve = int(budget * 0.35)
        protected = [e for e in durable if e["type"] == "FAIL" or (e["type"] == "OBSERVED" and _is_concrete(e["content"]))]
        r_used = 0
        for e in sorted(protected, key=lambda x: -x["_last"]):
            if r_used + size(e) > reserve:
                continue
            r_used += size(e)
            kept.append(e)
            kept_ids.add(id(e))
        used = r_used
        for e in sorted(durable, key=lambda x: (_priority(x), -x["_last"])):
            if id(e) in kept_ids or used + size(e) > budget:
                continue
            used += size(e)
            kept.append(e)
            kept_ids.add(id(e))
        r_kept, r_used = [], 0
        for e in sorted(results, key=lambda x: -x["step"])[:max_results]:
            if r_used + size(e) > result_tokens * CHARS_PER_TOKEN:
                continue
            r_used += size(e)
            r_kept.append(e)
        kept.sort(key=lambda e: e["seq"])
        r_kept.sort(key=lambda e: e["seq"])
        return r_kept + kept

    def render(self, step: int) -> str:
        lines = [f"==== SHARED BOARD (all agents; snapshot at step {step}) ====", "[Task queue]"]
        if not self.items:
            lines.append("(queue not yet created — the first agent is decomposing the task)")
        else:
            done = {it["id"] for it in self.items if it["status"] == "done"}
            rows_open, rows_claimed, rows_done = [], [], []
            for it in self.items:
                deps = ", ".join(f"{d} {'done' if d in done else 'pending'}" for d in it["deps"])
                dep_s = f" (deps: {deps})" if it["deps"] else ""
                if it["status"] == "open":
                    ready = all(d in done for d in it["deps"])
                    tag = "open   " if ready else "blocked"
                    rows_open.append(f"[{it['id']}] {tag}| {it['subject']} — {it['detail']}{dep_s}")
                elif it["status"] == "claimed":
                    rows_claimed.append(f"[{it['id']}] claimed by a{it['holder']} | {it['subject']}")
                else:
                    rows_done.append(f"[{it['id']}] done by a{it['holder']} | {it['subject']}")
            block = rows_open + rows_claimed + rows_done
            while block and count_tokens("\n".join(block)) > 1500:
                if rows_done:
                    rows_done.pop(0)
                elif rows_claimed:
                    rows_claimed.pop(0)
                else:
                    rows_open.pop()
                block = rows_open + rows_claimed + rows_done
            lines.extend(block)
        sel = self._select()
        res = [e for e in sel if e["type"] == "RESULT"]
        dur = [e for e in sel if e["type"] != "RESULT"]
        if res:
            lines.append("[Results — candidate results proposed by agents; review BEFORE deciding to submit or to propose your own]")
            for e in res:
                att = f"   (attachment: {e['attachment']}, expand_note)" if e.get("attachment") else ""
                lines.append(f"[a{e['agent']}/RESULT +s{e['step']}] {e['content']}{att}")
        lines.append("[Shared lessons]")
        if not dur:
            lines.append("(no shared lessons yet)")
        for e in dur:
            suffix = f" (x{e['_dup']})" if e["_dup"] > 1 else ""
            lines.append(f"[a{e['agent']}/{e['type']} +s{e['step']}] {e['content']}{suffix}")
        lines.append(GUIDANCE)
        return "\n".join(lines)


class DeLMProtocol(Protocol):
    name = "delm"

    def __init__(self, gate_tools: bool = True, finalize: bool = False, **kw):
        super().__init__(**kw)
        self.gate_tools, self.finalize = gate_tools, finalize
        if not gate_tools:
            self.name = "delm_nogate"
        if finalize:
            self.name = "delm_finalize"

    def build(self, run) -> None:
        board = Board(lambda: run.step)
        run.state["board"] = board
        run.state["delm"] = {"decomposer": 0, "claims": {}, "answers": {}, "seq": 0, "render_log": []}
        task = run.task
        self.task_tool_names = {t.name for t in task.tools(run.example, run)}
        n, nf = run.cfg.n_agents, run.cfg.n_failing
        total = n + nf
        fx, ox = EXAMPLES.get(task.name, ("name and exact value", "tool output"))
        for i in range(total):
            submit = task.submit_tool(run.example, with_refs=False)
            role = DELM_COMMON.format(i=i, N=total, submit=submit.name, fact_examples=fx, observed_examples=ox)
            role += "\n\n" + (DELM_DECOMPOSER.format(N=total) if i == 0 else DELM_WORKER)
            system = task.system_prompt(run.example) + "\n\n" + role
            tools = task.tools(run.example, run) + self._tools(run) + [submit]
            run.add_agent(system, task.task_prompt(run.example, i, total, run.rng), tools,
                          failing=is_failing(run, i), role="decomposer" if i == 0 else "worker")

    def _note(self, run, agent_id: int, typ: str, content: str, **extra) -> dict:
        st = run.state["delm"]
        st["seq"] += 1
        e = {"seq": st["seq"], "agent": agent_id, "type": typ, "content": content, "step": run.step, **extra}
        run.state["board"].notes.append(e)
        run.state["board"].bump()
        return e

    def _tools(self, run) -> list[ToolSpec]:
        board: Board = run.state["board"]
        st = run.state["delm"]
        task = run.task
        props, _ = task.answer_properties()
        att_cap = task.msg_token_limit

        def add_tasks(ctx: AgentCtx, args: dict) -> ToolResult:
            if not board.handoff_done and ctx.agent_id != st["decomposer"]:
                return ToolResult("Only the first agent creates the initial items.", ok=False)
            items = args.get("items") or []
            if not isinstance(items, list) or not items:
                return ToolResult("Error: items must be a non-empty list.", ok=False)
            ids = {it["id"] for it in board.items}
            added, reused = [], []
            for it in items[:16]:
                subj = str(it.get("subject", "")).strip()[:80]
                if not subj:
                    continue
                norm = " ".join(subj.lower().split())
                prior = next((x for x in board.items if " ".join(x["subject"].lower().split()) == norm), None)
                if prior:
                    reused.append(prior["id"])
                    continue
                deps = [str(d) for d in (it.get("deps") or [])]
                bad = [d for d in deps if d not in ids and d not in {a["id"] for a in added}]
                if bad:
                    return ToolResult(f"Rejected: deps {bad} do not exist.", ok=False)
                new = {"id": f"T{len(board.items) + len(added) + 1}", "subject": subj,
                       "detail": str(it.get("detail", ""))[:300], "deps": deps, "status": "open", "holder": None}
                added.append(new)
            board.items.extend(added)
            board.handoff_done = True
            board.bump()
            return ToolResult(f"Added {[a['id'] for a in added]}" + (f"; reused {reused}" if reused else "") + ".")

        def claim(ctx: AgentCtx, args: dict) -> ToolResult:
            iid = str(args.get("id", "")).strip()
            it = next((x for x in board.items if x["id"] == iid), None)
            if it is None:
                return ToolResult(f"Error: no item {iid}.", ok=False)
            if st["claims"].get(ctx.agent_id):
                return ToolResult(f"Error: you already hold {st['claims'][ctx.agent_id]}; complete it first.", ok=False)
            if it["status"] != "open":
                return ToolResult(f"Item {iid} is {it['status']} (holder a{it['holder']}).", ok=False)
            done = {x["id"] for x in board.items if x["status"] == "done"}
            if not all(d in done for d in it["deps"]):
                return ToolResult(f"Item {iid} is blocked on {it['deps']}.", ok=False)
            it["status"], it["holder"] = "claimed", ctx.agent_id
            st["claims"][ctx.agent_id] = iid
            board.bump()
            return ToolResult(f"Claimed {iid}: {it['subject']} — {it['detail']}")

        def complete(ctx: AgentCtx, args: dict) -> ToolResult:
            iid = str(args.get("id", "")).strip()
            if st["claims"].get(ctx.agent_id) != iid:
                return ToolResult(f"Error: you do not hold {iid}.", ok=False)
            summary = str(args.get("summary", "")).strip()[:MAX_RESULT_CHARS]
            if not summary:
                return ToolResult("Error: summary must not be empty.", ok=False)
            it = next(x for x in board.items if x["id"] == iid)
            it["status"] = "done"
            st["claims"].pop(ctx.agent_id, None)
            att_id = None
            if args.get("attachment"):
                att_id = f"N{len(board.attachments) + 1}"
                board.attachments[att_id] = truncate_to_tokens(str(args["attachment"]), att_cap)
            self._note(run, ctx.agent_id, "RESULT", summary, attachment=att_id, item=iid)
            if isinstance(args.get("answer"), dict):
                st["answers"].setdefault(ctx.agent_id, []).append(args["answer"])
            elif args.get("attachment") and hasattr(task, "answer_from_attachment"):
                st["answers"].setdefault(ctx.agent_id, []).append(task.answer_from_attachment(str(args["attachment"])))
            return ToolResult(f"Completed {iid}." + (f" Attachment stored as {att_id}." if att_id else ""))

        def note(ctx: AgentCtx, args: dict) -> ToolResult:
            entries = args.get("entries") or []
            if not isinstance(entries, list) or not 1 <= len(entries) <= 3:
                return ToolResult("Error: write 1-3 entries.", ok=False)
            written, errs = 0, []
            for e in entries:
                typ, content = str(e.get("type", "")).upper(), " ".join(str(e.get("content", "")).split())
                if typ not in NOTE_TYPES:
                    errs.append(f"invalid type {typ}")
                    continue
                if not content or len(content) > MAX_NOTE_CHARS:
                    errs.append(f"{typ} content must be 1-{MAX_NOTE_CHARS} chars")
                    continue
                if typ == "TRIED" and _is_routine_tried(content):
                    continue
                self._note(run, ctx.agent_id, typ, content)
                written += 1
            return ToolResult(f"Wrote {written} entries." + (f" Rejected: {'; '.join(errs)}." if errs else ""),
                              ok=not errs)

        def expand(ctx: AgentCtx, args: dict) -> ToolResult:
            k = str(args.get("id", "")).strip()
            return ToolResult(board.attachments[k]) if k in board.attachments else ToolResult(
                f"Error: no attachment {k}.", ok=False)

        def wait(ctx: AgentCtx, args: dict) -> ToolResult:
            agent = next(a for a in run.agents if a.id == ctx.agent_id)
            v0 = board.version
            agent.wait(lambda: board.version != v0, until_step=run.step + 6)
            return ToolResult("Waiting for the board to change (at most 5 steps).")

        S = lambda **p: {"type": "object", "properties": p, "required": [k for k in p if k not in ("answer", "attachment", "deps")]}
        return [
            ToolSpec("add_tasks", "Publish subtasks of the original task to the shared TASK QUEUE. Before the first handoff only the first agent may call this.",
                     {"type": "object", "properties": {"items": {"type": "array", "maxItems": 16, "items": {
                         "type": "object", "properties": {"subject": STR, "detail": STR,
                                                          "deps": {"type": "array", "items": STR}},
                         "required": ["subject", "detail"]}}}, "required": ["items"]}, add_tasks),
            ToolSpec("claim_task", "Claim an open, ready item. First claim wins; you may hold one unfinished claim.",
                     S(id=STR), claim),
            ToolSpec("complete_task", "Finish your claim: result, limitations, checks already obtained.",
                     S(id=STR, summary={**STR, "description": "RESULT <=300 chars: result=... | evidence=... | risk=..."},
                       answer={"type": "object", "properties": props, "description": "optional candidate answer"},
                       attachment={**STR, "description": "optional; fetched by others via expand_note"}), complete),
            ToolSpec("note", "Write 1-3 short typed entries to SHARED LESSONS.",
                     {"type": "object", "properties": {"entries": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                         "type": "object", "properties": {"type": {"type": "string", "enum": list(NOTE_TYPES)},
                                                          "content": STR}, "required": ["type", "content"]}}},
                      "required": ["entries"]}, note),
            ToolSpec("expand_note", "Show the attachment of a completed item.", S(id=STR), expand),
            ToolSpec("wait_board", "Wait until the board changes (new item, completion, note) or 5 steps pass. Use only when you need a specific input a peer is producing.",
                     {"type": "object", "properties": {}}, wait),
        ]

    def gate(self, run, agent, tool_name: str) -> str | None:
        st, board = run.state["delm"], run.state["board"]
        if tool_name == agent.submit_name and not board.handoff_done and agent.id != st["decomposer"]:
            return "The task queue does not exist yet; call wait_board until the first agent publishes items."
        if self.gate_tools and tool_name in self.task_tool_names and not st["claims"].get(agent.id):
            return "No claimed subtask. Claim an open item first (claim_task) or wait_board."
        return None

    def live_block(self, run, agent, step: int) -> str | None:
        text = run.state["board"].render(step)
        run.state["delm"]["render_log"].append({"step": step, "agent": agent.id, "tokens": count_tokens(text),
                                                "entries_total": len(run.state["board"].notes)})
        return text

    def on_agent_done(self, run, agent) -> None:
        st, board = run.state["delm"], run.state["board"]
        iid = st["claims"].pop(agent.id, None)
        if iid:
            it = next(x for x in board.items if x["id"] == iid)
            it["status"], it["holder"] = "open", None
            self._note(run, agent.id, "FAIL", f"{iid} released by stopped agent")
        if not board.handoff_done and agent.id == st["decomposer"] and not agent.failing:
            nxt = next((a.id for a in run.agents if not a.done and a.id > agent.id), None)
            if nxt is not None:
                st["decomposer"] = nxt

    def needs_final(self, run, agent) -> bool:
        if self.finalize:
            return not run.state["delm"]["answers"].get(agent.id)
        return False

    def collect(self, run) -> list[dict]:
        out = []
        for a in run.agents:
            final = a.final
            if final is None:
                final = run.task.recorded_answer(run.state["delm"]["answers"].get(a.id, []))
            out.append({"agent": a.id, "failing": a.failing, "role": a.role, "final": final,
                        "forced": a.final is None})
        return out

    def dump(self, run) -> dict:
        board = run.state["board"]
        rl = run.state["delm"]["render_log"]
        return {"items": board.items, "notes": board.notes, "n_attachments": len(board.attachments),
                "handoff_done": board.handoff_done,
                "board_tokens_mean": sum(r["tokens"] for r in rl) / max(1, len(rl)), "renders": len(rl)}
