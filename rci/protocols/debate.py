"""Multi-agent debate (all-to-all message passing) on the shared tool-using harness, with its evidence variants."""
from __future__ import annotations

import os
import re

from rci.agents.tokens import count_tokens, truncate_to_tokens
from rci.protocols.base import Protocol
from rci.protocols.team_notes import load_team_once, team_block

DEBATE_INJECT = """These are the recent work records from other agents (everything they did since your last turn):

{blocks}

Using the work from other agents as additional information, continue solving the task. Examine your own work and that of the other agents step by step; reuse, verify or correct it as appropriate. When you are confident in your answer, call `{submit}`."""

HOLD = """Not submitted: in this debate everyone answers after {rounds} rounds, and this is round {k}. Write your reasoning and your current answer for the others in plain text; you will read theirs before your next turn."""
HOLD_FACTS = """Not submitted: in this debate everyone answers after {rounds} rounds, and this is round {k}. Write, for the others in plain text, every fact you hold and what follows from it, but not which option you favour; you will read theirs before your next turn."""
FACTS_RULE = """In this debate the others read everything you write. Share the facts you hold and what follows from them, but do not say which option you favour: everyone decides on their own after the last round."""
FORWARD_RULE = """In this debate the others read everything you write. Post every fact you were given, word for word, and nothing else: no analysis, no interpretation, no emphasis, no conclusion, and not which option you favour. Everyone decides on their own after the last round."""
HOLD_FORWARD = """Not submitted: in this debate everyone answers after {rounds} rounds, and this is round {k}. Post, for the others, every fact you were given, word for word, and nothing else; you will read theirs before your next turn."""
SELF_INJECT = """This is your own work from round 1 (everything you did in that round):

```{text}```

Using this work as additional information, continue solving the task. Examine it step by step; reuse, verify or correct it as appropriate. When you are confident in your answer, call `{submit}`."""
HOLD_SELF = """Not submitted: you answer after {rounds} rounds, and this is round {k}. Write your reasoning and your current answer in plain text; you will re-read your own round-1 work before your next turn."""
WITHHELD = "answer withheld"
TOOL_CALL_TEXT = re.compile(r"<tool_call>.*?(</tool_call>|$)"
                            r"|(<\|start\|>assistant)?<\|channel\|>commentary to=functions\..*?(<\|call\|>|$)", re.S)
TENTATIVE = "\x00tentative"


class DebateProtocol(Protocol):
    """Debate-R, Fact forwarding, Shared context, Self-Refine and the supervisor-written team (`team_file`)."""
    name = "debate"

    def __init__(self, content_cap: int = 1024, args_cap: int = 256, result_cap: int = 512,
                 record_cap: int = 2048, inject_budget: int = 32768, include_results: bool = True, rounds: int = 1,
                 views: bool = True, forward_facts: bool = False, shared_context: bool = False, self_refine: bool = False,
                 team_file: str = "", **kw):
        super().__init__(**kw)
        if self_refine and (int(rounds) != 2 or not views or forward_facts or shared_context):
            raise ValueError("self_refine needs rounds=2, views=True, forward_facts=False and shared_context=False")
        self.team_file = os.path.expandvars(str(team_file or ""))
        self._team, self._team_sha = None, None
        if self.team_file:
            if forward_facts or shared_context or self_refine or not views or not include_results:
                raise ValueError("team_file goes with views=True only: not with forward_facts, shared_context, "
                                 "self_refine, views=False or include_results=False")
            self._team, self._team_sha = load_team_once(self.team_file)
            if self._team["rounds"] != max(1, int(rounds)):
                raise ValueError(f"team_file {self.team_file}: written for {self._team['rounds']} rounds, "
                                 f"the protocol runs {max(1, int(rounds))}")
        self.self_refine = bool(self_refine)
        self.caps = dict(content=content_cap, args=args_cap, result=result_cap, record=record_cap)
        self.inject_budget = inject_budget
        self.include_results = include_results
        self.rounds = max(1, int(rounds))
        self.forward_facts = bool(forward_facts)
        self.shared_context = bool(shared_context)
        self.views = bool(views) and not self.forward_facts
        if not self.views and self.rounds < 2:
            raise ValueError("views=False / forward_facts need rounds >= 2: with one round every member submits before "
                             "reading anyone")
        if not include_results:
            self.name = "debate_content_only"
        if self.rounds > 1:
            self.name = f"{self.name}_r{self.rounds}"
        if self.shared_context:
            self.name = f"{self.name}_shared_context"
        if self.forward_facts:
            self.name = f"{self.name}_forward_facts"
        elif not self.views:
            self.name = f"{self.name}_facts"
        if self.self_refine:
            self.name = f"{self.name}_self_refine"

    def method_prompt(self, run, agent_id: int) -> str:
        return FORWARD_RULE if self.forward_facts else "" if self.views else FACTS_RULE

    def notes_for(self, agent_id: int) -> str:
        """The member's team block ("" without a team file)."""
        if self._team is None:
            return ""
        return team_block(self._team["shared"], self._team["cards"][agent_id], agent_id)

    def build_system(self, run, agent_id: int, task_system: str) -> str | None:
        """With team notes: the task's system prompt, then the member's block."""
        if self._team is None:
            return None
        return task_system + self.notes_for(agent_id)

    def build(self, run) -> None:
        if self._team is not None:
            n = run.cfg.n_agents + run.cfg.n_failing
            if len(self._team["cards"]) != n:
                raise ValueError(f"team_file {self.team_file}: {len(self._team['cards'])} cards for {n} members")
        run.state["debate_records"] = []
        run.state["debate_last_turn"] = {}
        run.state["debate_inj_log"] = []
        run.state["debate_revealed"] = []
        super().build(run)

    def _parts(self, run, agent) -> dict:
        if agent.failing:
            return {"fault": agent.last_output}
        results = {m.get("tool_call_id"): str(m.get("content", "")) for m in agent.last_turn if m["role"] == "tool"}
        content, calls = "", []
        for m in agent.last_turn:
            if m["role"] != "assistant":
                continue
            content = (m.get("content") or "").strip()
            if not self.views:
                content = TOOL_CALL_TEXT.sub("", content).strip()
            for tc in m.get("tool_calls", []):
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                if name == agent.submit_name:
                    res = None if agent.done else TENTATIVE
                    if not self.views:
                        args = WITHHELD
                else:
                    res = results.get(tc["id"], "")
                calls.append((name, args, res))
        out = {"content": content, "calls": calls}
        if self.shared_context and run is not None and agent.id not in run.state["debate_revealed"]:
            run.state["debate_revealed"].append(agent.id)
            n = run.cfg.n_agents + run.cfg.n_failing
            out["revealed"] = run.task.facts(run.example, agent.id, n)
        return out

    def _render(self, parts: dict, content_cap: int, args_cap: int, result_cap: int) -> str:
        if "fault" in parts:
            return parts["fault"]
        out = []
        if parts.get("revealed"):
            out.append("[every fact this member was given]\n" + "\n".join(f"- {x}" for x in parts["revealed"]))
        if parts["content"]:
            out.append("[assistant] " + truncate_to_tokens(parts["content"], content_cap))
        for name, args, res in parts["calls"]:
            out.append(f"-> {name}({truncate_to_tokens(args, args_cap)})")
            if res is None:
                out.append("[submitted]")
            elif res == TENTATIVE:
                out.append("[tentative answer, not submitted yet]")
            elif self.include_results:
                out.append("<- result: " + truncate_to_tokens(res, result_cap))
        return "\n".join(out)

    def _fit(self, parts: dict, budget: int) -> str:
        c, a, r = self.caps["content"], self.caps["args"], self.caps["result"]
        budget = min(budget, self.caps["record"])
        text = self._render(parts, c, a, r)
        for which in ("result", "args", "content"):
            while count_tokens(text) > budget:
                if which == "result" and r > 16:
                    r //= 2
                elif which == "args" and a > 16:
                    a //= 2
                elif which == "content" and c > 16:
                    c //= 2
                else:
                    break
                text = self._render(parts, c, a, r)
        return text if count_tokens(text) <= budget else truncate_to_tokens(text, budget)

    def on_turn_end(self, run, agent, step: int) -> None:
        run.state["debate_last_turn"][agent.id] = step
        parts = self._parts(run, agent)
        text = self._fit(parts, self.caps["record"])
        if text:
            run.state["debate_records"].append((step, agent.id, parts, count_tokens(text)))

    def injection(self, run, agent, step: int) -> str | None:
        if self.self_refine:
            return self._self_injection(run, agent, step)
        if len(run.agents) <= 1:
            return None
        since = run.state["debate_last_turn"].get(agent.id, -1)
        recs = [r for r in run.state["debate_records"] if r[1] != agent.id and since <= r[0] < step]
        if not recs:
            return None
        total = sum(r[3] for r in recs)
        truncated = total > self.inject_budget
        per = max(32, self.inject_budget // len(recs)) if truncated else self.caps["record"]
        rendered = [(s, j, self._fit(parts, per)) for s, j, parts, _ in recs]
        blocks = "\n\n".join(f"One agent's work [agent {j}, step {s}]: ```{t}```" for s, j, t in rendered)
        run.state["debate_inj_log"].append({"step": step, "agent": agent.id, "records": len(recs),
                                         "tokens": sum(count_tokens(t) for _, _, t in rendered),
                                         "truncated": truncated})
        return DEBATE_INJECT.format(blocks=blocks, submit=agent.submit_name)

    def _self_injection(self, run, agent, step: int) -> str | None:
        log = run.state["debate_inj_log"]
        if any(x["agent"] == agent.id for x in log):
            return None
        recs = [r for r in run.state["debate_records"] if r[1] == agent.id and r[0] < step]
        if not recs:
            return None
        total = sum(r[3] for r in recs)
        truncated = total > self.inject_budget
        per = max(32, self.inject_budget // len(recs)) if truncated else self.caps["record"]
        rendered = [self._fit(parts, per) for _, _, parts, _ in recs]
        log.append({"step": step, "agent": agent.id, "records": len(recs),
                    "tokens": sum(count_tokens(t) for t in rendered), "truncated": truncated, "self": True})
        return SELF_INJECT.format(text="\n\n".join(rendered), submit=agent.submit_name)

    def gate(self, run, agent, tool_name: str) -> str | None:
        """Hold a submission until the member has read the others rounds-1 times, or until no one is left to read."""
        if self.self_refine:
            if tool_name != agent.submit_name:
                return None
            read = sum(1 for x in run.state["debate_inj_log"] if x["agent"] == agent.id and x.get("self"))
            return None if read >= self.rounds - 1 else HOLD_SELF.format(rounds=self.rounds, k=read + 1)
        if self.rounds <= 1 or tool_name != agent.submit_name or len(run.agents) <= 1:
            return None
        read = sum(1 for x in run.state["debate_inj_log"] if x["agent"] == agent.id)
        if read >= self.rounds - 1 or all(a.done or a.failing for a in run.agents if a.id != agent.id):
            return None
        hold = HOLD_FORWARD if self.forward_facts else HOLD if self.views else HOLD_FACTS
        return hold.format(rounds=self.rounds, k=read + 1)

    def dump(self, run) -> dict:
        log = run.state["debate_inj_log"]
        out = {"rounds": self.rounds, "views": self.views, "forward_facts": self.forward_facts,
               "shared_context": self.shared_context, "n_records": len(run.state["debate_records"]),
               "inj_tokens_total": sum(x["tokens"] for x in log),
               "inj_truncated": sum(1 for x in log if x["truncated"]), "inj_log": log}
        if self.self_refine:
            out["self_refine"] = True
        if self._team is not None:
            blocks = [self.notes_for(a.id) for a in run.agents]
            out.update({"notes_file": self.team_file, "notes_sha256": self._team_sha,
                        "notes_version": self._team["version"], "notes_blocks": blocks,
                        "notes_in_prompt": [(a.system_prompt or "").count(b) == 1 for a, b in zip(run.agents, blocks)]})
        return out
