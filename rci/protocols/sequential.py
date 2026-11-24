"""Single-agent references: the sequential lone agent, Sequential-k attempts, the sequential map and the oracle."""
from __future__ import annotations

import importlib

from rci.agents.llm import ContextTooLong
from rci.agents.tooling import AgentCtx, ToolResult, ToolSpec
from rci.protocols.base import Protocol
from rci.protocols.docsweep_board import (DIGEST_RULES, LONE_BLOCKS, MAP_INTEGRATOR_BLOCKS, agent_cfg, board_block,
                                         covered_count, digest_tool, ds_confirm, ds_hold, ds_init, ds_offer, ds_result,
                                         ds_tool, n_docs)
from rci.tasks.base import stated

SOLO_RULES = ("You work alone, with {actions} actions and {steps} rounds. You can submit your answer only when both of "
              "these hold: you have opened every document (in rounds before the one in which you submit), and you have "
              "used most of your budget, at least 90% of it, which the host tracks for you. Until then a submission is "
              "held, and a held submission costs no action. When you reach your action limit or your round limit, your "
              "answer is submitted for you.")
SOLO_HOLD = "You still have budget left: reopen and check documents and revise your notes before you submit."
SPEND_SHARE = 0.9

SEQ_SHOWN_HEAD = "Your earlier submissions on this question, in order:"
SEQ_LINE = "- Attempt {i}: answer {answer}, confidence {confidence}"
SEQ_ATTEMPT = "This is attempt {j} of {n}."
SEQ_TURNS = 24

MAP_ROLES = ("worker1", "worker2", "worker3", "integrator")
MAP_ACTIONS, MAP_TURNS = 12, 24
MAP_WORKER = ("You are worker {w} of 3 in a fixed program: three workers, one after another, each in a fresh context, then "
              "an integrator that answers the question from the workers' verified digests. Your assigned documents are "
              "{docs}. You may open any document, but your job is to read your assigned documents and post verified "
              "digests of them with `post_digest`, quoting word for word every passage the question may need. The "
              "integrator sees only the verified digests; it does not see your notes or your reasoning, and the workers "
              "do not see each other's work. " + DIGEST_RULES + " Call `done` when you have finished.")
MAP_INTEGRATOR = ("You are the integrator of a fixed program: before you, three workers, one after another and each in a "
                  "fresh context, read about a third of the documents each and posted verified digests (quotes copied "
                  "word for word from documents they opened, checked by the host). You receive every verified digest "
                  "before your first call. Answer the question from them; you may also open or search any document "
                  "yourself to fill a gap or check a value.")
DONE_TEXT = "Done: your digests are on the board for the integrator."

ORACLE_PROMPT_CAP = 36864


class _CorpusOverflowFallback(Exception):
    pass


def corpus_overflow_class(task):
    """The task's CorpusOverflow exception (defined in the DocSweep task module)."""
    cls = getattr(task, "CorpusOverflow", None) or getattr(importlib.import_module(type(task).__module__),
                                                           "CorpusOverflow", None)
    return cls or _CorpusOverflowFallback


def single_member(run, name: str) -> None:
    """These references are one agent per run (records at N1): refuse any other group size."""
    if run.cfg.n_agents != 1 or run.cfg.n_failing:
        raise ValueError(f"{name} runs one agent: use --n-agents 1 and no failing agents")


class Sequential(Protocol):
    """The DocSweep lone agent with m members' budget, or `attempts` fresh contexts in sequence (Sequential-k)."""
    name = "sequential"
    note_cap = 16800

    def __init__(self, m: int | None = None, c1_bar: float = 10000.0, c1_gpu_bar: float | None = None,
                 rho: float | None = None, attempts: int = 0):
        super().__init__()
        self.attempts = int(attempts)
        if self.attempts < 0:
            raise ValueError("attempts must be >= 0")
        if self.attempts > 0:
            self.name = "sequential_attempts"
        elif m is None or int(m) < 1:
            raise ValueError("Sequential needs m >= 1 (or attempts > 0)")
        self.m = int(m) if m is not None else None
        self.c1_bar = float(c1_bar)
        self.c1_gpu_bar = None if c1_gpu_bar is None else float(c1_gpu_bar)
        self.rho = None if rho is None else float(rho)

    def build(self, run) -> None:
        single_member(run, self.name)
        if self.attempts > 0:
            if run.cfg.max_steps < SEQ_TURNS * self.attempts:
                raise ValueError(f"seq{self.attempts} needs max_steps >= {SEQ_TURNS * self.attempts}")
            run.state["seq_shown"] = []
            self.add_attempt(run, 0)
            return
        m = self.m
        if run.cfg.max_actions != 12 * m or run.cfg.max_steps != 24 * m:
            raise ValueError(f"S_{m} needs --max-actions {12 * m} and --max-steps {24 * m}, got "
                             f"{run.cfg.max_actions} and {run.cfg.max_steps}")
        if self.c1_gpu_bar is not None and self.rho is None:
            raise ValueError("c1_gpu_bar needs rho (the GPU unit is completion + rho * new prompt tokens)")
        task, ex = run.task, run.example
        run.state.setdefault("holds", [])
        system = (task.system_prompt(ex, kind="solo", blocks=LONE_BLOCKS, actions=12 * m, steps=24 * m) + "\n\n"
                  + SOLO_RULES.format(actions=12 * m, steps=24 * m))
        tools = task.tools(ex, run) + [task.submit_tool(ex, with_refs=False)]
        run.add_agent(system, task.task_prompt(ex, 0, 1, run.rng), tools, role="solo")

    def spend(self, agent) -> tuple[int, float | None]:
        comp = sum(int(c.completion_tokens or 0) for c in agent.calls)
        if self.rho is None:
            return comp, None
        gpu = sum(int(c.completion_tokens or 0) + self.rho * (getattr(c, "new_prompt_tokens", None) or 0)
                  for c in agent.calls)
        return comp, gpu

    def floor(self) -> tuple[float, float | None]:
        """The spend floor: 0.9*m*c1_bar completion tokens and, when set, 0.9*m*c1_gpu_bar GPU-unit tokens."""
        return (SPEND_SHARE * self.m * self.c1_bar,
                None if self.c1_gpu_bar is None else SPEND_SHARE * self.m * self.c1_gpu_bar)

    def gate(self, run, agent, tool_name: str) -> str | None:
        if self.attempts > 0 or tool_name != agent.submit_name:
            return None
        docs_ok = bool(run.task.holds_every_fact(run.example, run, agent.id))
        comp, gpu = self.spend(agent)
        f_comp, f_gpu = self.floor()
        spend_ok = comp >= f_comp and (f_gpu is None or (gpu is not None and gpu >= f_gpu))
        if docs_ok and spend_ok:
            return None
        reason = "both" if not docs_ok and not spend_ok else "docs" if not docs_ok else "spend"
        ds_hold(run, agent, "solo", tool_name, reason=reason, completion=comp, gpu=gpu)
        return Protocol.Hold(SOLO_HOLD)

    def defer_final(self, run, agent) -> bool:
        """At either cap the hold lifts: the host forces the submission."""
        return False

    def needs_final(self, run, agent) -> bool:
        return True

    def add_attempt(self, run, j: int) -> None:
        task, ex = run.task, run.example
        shown = []
        for i, a in enumerate(run.agents[:j]):
            f = a.final if isinstance(a.final, dict) else None
            shown.append({"attempt": i + 1, "answer": (f or {}).get("answer"), "confidence": stated(f)})
        prompt = task.task_prompt(ex, 0, 1, run.rng)
        if shown:
            lines = [SEQ_LINE.format(i=s["attempt"], answer=self.show(s["answer"]),
                                     confidence="not stated" if s["confidence"] is None else s["confidence"])
                     for s in shown]
            prompt += "\n\n" + SEQ_SHOWN_HEAD + "\n" + "\n".join(lines)
        prompt += "\n\n" + SEQ_ATTEMPT.format(j=j + 1, n=self.attempts)
        run.state["seq_shown"].append(shown)
        tools = task.tools(ex, run) + [task.submit_tool(ex, with_refs=False)]
        run.add_agent(task.system_prompt(ex), prompt, tools, role=f"attempt{j + 1}",
                      cfg=agent_cfg(run, max_turns=SEQ_TURNS))

    @staticmethod
    def show(answer) -> str:
        if answer is None:
            return "none"
        if isinstance(answer, str):
            return answer
        import json
        return json.dumps(answer, ensure_ascii=False)

    def on_agent_done(self, run, agent) -> None:
        """Sequential-k: the next attempt starts once this one is done (while a step is left for it)."""
        if self.attempts > 0:
            j = len(run.agents)
            if agent.id == j - 1 and j < self.attempts and run.step < run.cfg.max_steps - 1:
                self.add_attempt(run, j)
            return
        comp, gpu = self.spend(agent)
        run.state["solo_at_submit"] = {"completion": comp, "gpu": gpu, "covered": covered_count(run, agent.id)}

    def finished(self, run) -> bool:
        if self.attempts > 0:
            return len(run.agents) >= self.attempts and all(a.done for a in run.agents)
        return all(a.done for a in run.agents)

    def dump(self, run) -> dict:
        if self.attempts > 0:
            return {"attempts": self.attempts, "shown": run.state.get("seq_shown", [])}
        f_comp, f_gpu = self.floor()
        return {"m": self.m, "c1_bar": self.c1_bar, "c1_gpu_bar": self.c1_gpu_bar, "rho": self.rho,
                "holds": run.state.get("holds", []), "floor": {"completion": f_comp, "gpu": f_gpu},
                "at_submit": run.state.get("solo_at_submit")}


class SequentialMap(Protocol):
    name = "sequential_map"
    note_cap = 6000

    def __init__(self):
        super().__init__()

    def build(self, run) -> None:
        single_member(run, self.name)
        if run.cfg.max_steps < len(MAP_ROLES) * MAP_TURNS:
            raise ValueError(f"S_map needs max_steps >= {len(MAP_ROLES) * MAP_TURNS}, got {run.cfg.max_steps}")
        ds_init(run, len(MAP_ROLES))
        k = n_docs(run)
        run.state["map_assignment"] = {MAP_ROLES[w]: [d for d in range(1, k + 1) if (d - 1) % 3 == w]
                                       for w in range(3)}
        self.add_role(run, 0)

    def add_role(self, run, j: int) -> None:
        task, ex = run.task, run.example
        role = MAP_ROLES[j]
        cfg = agent_cfg(run, max_actions=MAP_ACTIONS, max_turns=MAP_TURNS)
        prompt = task.task_prompt(ex, 0, 1, run.rng)
        if role == "integrator":
            system = (task.system_prompt(ex, kind="member", blocks=MAP_INTEGRATOR_BLOCKS, actions=MAP_ACTIONS,
                                         steps=MAP_TURNS) + "\n\n" + MAP_INTEGRATOR)
            tools = task.tools(ex, run) + [task.submit_tool(ex, with_refs=False)]
            agent = run.add_agent(system, prompt, tools, role=role, cfg=cfg)
            run.state["ds_pending"][agent.id] = [it["id"] for it in sorted(run.state["ds_board"].values(),
                                                                          key=lambda it: it["seq"])]
            return
        docs = run.state["map_assignment"][role]
        system = (task.system_prompt(ex, kind="member", blocks=LONE_BLOCKS, actions=MAP_ACTIONS, steps=MAP_TURNS)
                  + "\n\n" + MAP_WORKER.format(w=j + 1, docs=", ".join(str(d) for d in docs)))
        tools = (task.tools(ex, run) + [digest_tool(run, lambda r, me: (), "")]
                 + [self.done_tool()])
        run.add_agent(system, prompt, tools, role=role, cfg=cfg)

    @staticmethod
    def done_tool() -> ToolSpec:
        """A worker's terminal tool: `done` ({} -> final {"done": True})."""
        def done(ctx: AgentCtx, args: dict) -> ToolResult:
            return ds_result(DONE_TEXT, final={"done": True})
        return ds_tool("done", "End your work (one action). Your verified digests stay on the board for the integrator.",
                       {"type": "object", "properties": {}}, done, terminal=True)

    def on_agent_done(self, run, agent) -> None:
        """The next role starts once this one is done (while a step is left for it)."""
        j = len(run.agents)
        if agent.id == j - 1 and j < len(MAP_ROLES) and run.step < run.cfg.max_steps - 1:
            self.add_role(run, j)

    def finished(self, run) -> bool:
        return len(run.agents) >= len(MAP_ROLES) and all(a.done for a in run.agents)

    def injection(self, run, agent, step: int) -> str | None:
        if agent.role != "integrator":
            return None
        return ds_offer(run, agent, head="Verified digests posted by the workers:")

    def on_prompt(self, run, agent, call: int, via: str) -> None:
        if agent.role != "integrator":
            run.state["ds_offer"].pop(agent.id, None)
            return
        rebuilt = [it["id"] for it in run.state["ds_board"].values()] if via == "rebuilt" else []
        ds_confirm(run, agent, call, via, rebuilt)

    def memory_blocks(self, run, agent) -> list[tuple[str, str]]:
        return [board_block(run)] if agent.role == "integrator" else []

    def collect(self, run) -> list[dict]:
        return [{"agent": a.id, "failing": a.failing, "role": a.role, "final": a.final, "forced": a.final_forced}
                for a in run.agents if a.role == "integrator"]

    def dump(self, run) -> dict:
        st = run.state
        return {"roles": [a.role for a in run.agents], "assignment": st.get("map_assignment", {}),
                "items": st["ds_items"], "board": {d: st["ds_board"][d] for d in sorted(st["ds_board"])},
                "refusals": st["ds_refusals"]}


class _OverflowGuard:
    def __init__(self, llm, run):
        self._llm, self._run = llm, run

    async def chat(self, *a, **kw):
        try:
            return await self._llm.chat(*a, **kw)
        except ContextTooLong:
            self._run.state["oracle_overflow"] = True
            raise

    def __getattr__(self, name):
        return getattr(self._llm, name)


class Oracle(Protocol):
    name = "oracle"

    def __init__(self):
        super().__init__()

    def build(self, run) -> None:
        single_member(run, self.name)
        task, ex = run.task, run.example
        cfg = agent_cfg(run, compact_at=0, status_line=False, max_actions=3)
        agent = run.add_agent(task.system_prompt(ex, kind="oracle"), task.oracle_prompt(ex),
                              [task.submit_tool(ex, with_refs=False)], role="oracle", cfg=cfg)
        agent.llm = _OverflowGuard(agent.llm, run)
        ids = task.chat_tokenizer().render(agent.messages(), tools=agent.tool_schemas(),
                                           enable_thinking=cfg.enable_thinking)
        run.state["oracle_prompt_tokens"] = len(ids)
        if len(ids) > ORACLE_PROMPT_CAP:
            raise corpus_overflow_class(task)(
                f"oracle prompt of {ex.get('id')} renders to {len(ids)} tokens > {ORACLE_PROMPT_CAP}")

    def collect(self, run) -> list[dict]:
        """The oracle's final, unless the server refused its prompt (CorpusOverflow: no record is written)."""
        if run.state.get("oracle_overflow"):
            raise corpus_overflow_class(run.task)(f"oracle context overflow on {run.example.get('id')}")
        return Protocol.collect(self, run)

    def dump(self, run) -> dict:
        return {"prompt_tokens_rendered": run.state.get("oracle_prompt_tokens")}
