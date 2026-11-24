"""Task interface shared by every protocol."""
from __future__ import annotations

from typing import Callable

from rci.agents.tooling import STR, STR_LIST, AgentCtx, ToolResult, ToolSpec

SUBMIT_NAME = "submit_answer"
CONFIDENCE_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
CONFIDENCE_DESCRIPTION = ("How likely this answer is to be correct, one of 0, 0.2, 0.4, 0.6, 0.8. The group commits "
                          "to the answer its members back with the most confidence, so give your honest estimate.")
STAKE_DESCRIPTIONS = {"confidence": CONFIDENCE_DESCRIPTION}


def norm_confidence(x) -> float | None:
    """Nearest confidence level; None when the value is missing or not a number."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return min(CONFIDENCE_LEVELS, key=lambda b: abs(b - v))


def stated(final: dict | None) -> float | None:
    """The confidence a submission states, or None."""
    if not isinstance(final, dict) or not final:
        return None
    return final.get("confidence")


def stake(final: dict | None) -> float:
    """stated() with a missing confidence counted as 0, as every commitment rule counts it."""
    return float(stated(final) or 0.0)


RefValidator = Callable[[int, str, list[str]], "tuple[list[str], str | None]"]


class Task:
    name = "task"
    msg_token_limit = 256
    max_actions = 300
    max_steps = 400
    summarize = True
    stake_field: str | None = None
    deliberate_first = False
    deliberate_gen_limit: int | None = None

    def deliberates(self, example: dict) -> bool:
        """Whether the members' first turn on this example is a tool-free deliberation (default: deliberate_first)."""
        return bool(self.deliberate_first)

    def examples(self, limit: int | None = None) -> list[dict]:
        raise NotImplementedError

    def system_prompt(self, example: dict) -> str:
        raise NotImplementedError

    def task_prompt(self, example: dict, agent_id: int, n_agents: int, rng) -> str:
        raise NotImplementedError

    def tools(self, example: dict, run) -> list[ToolSpec]:
        return []

    def blackboard_tools(self, example: dict, run, datastore) -> list[ToolSpec]:
        """Extra tools only available when members share a blackboard."""
        return []

    def resource_exists(self, example: dict, run, ref: str) -> bool:
        return False

    def publish_validator(self, example: dict, run):
        return None

    def describe_message(self, sub_task: str, artifact: str) -> str:
        """A message's header for a reader who has not read it (the choice condition): what kind of message, no content."""
        return "a message"

    def answer_view(self, example: dict, text: str) -> str | None:
        """A certified message's conclusion without the facts it quotes (broadcast="answer")."""
        return None

    def certified(self, sub_task: str) -> bool:
        """Whether an accepted message is vouched for by the task's own gate."""
        return False

    def holds_every_fact(self, example: dict, run, author: int) -> bool:
        """Whether `author` has been shown every member's private information; tasks without private shares: always."""
        return True

    def answer_properties(self) -> tuple[dict, list[str]]:
        return {"answer": {**STR, "description": "Your final answer."}}, ["answer"]

    def submit_description(self) -> str:
        return "Submit your final answer. This ends your work on the task."

    def submit_tool(self, example: dict, with_refs: bool, validator: RefValidator | None = None,
                    repair=None) -> ToolSpec:
        props, req = self.answer_properties()
        props = dict(props)
        req = list(req)
        if with_refs:
            props["refs"] = {**STR_LIST, "description": "Messages and resources your answer rests on."}
            props["rationale"] = {**STR, "description": "Brief: how you got the answer, NAMING each ref and what it contributed."}
            req += ["refs", "rationale"]
        st = self.stake_field
        if st:
            props[st] = {"type": "number", "enum": list(CONFIDENCE_LEVELS), "description": STAKE_DESCRIPTIONS[st]}
            req.append(st)

        def fn(ctx: AgentCtx, args: dict) -> ToolResult:
            missing = [k for k in req if k not in args]
            relaxable = ({"refs", "rationale"} if with_refs else set()) | ({st} if st else set())
            if missing and ctx.state.get("final_last_attempt") and set(missing) <= relaxable:
                defaults = {"refs": [], "rationale": "", st: None}
                args = {**args, **{k: args.get(k) or defaults[k] for k in missing}}
                missing = []
            if missing:
                return ToolResult(f"Error: missing fields {missing}.", ok=False)
            final = {k: args.get(k) for k in props}
            if st:
                final[st] = norm_confidence(args.get(st))
            if with_refs and validator is not None:
                refs, err = validator(ctx.agent_id, str(args.get("rationale", "")), args.get("refs"))
                if err and repair is not None and ctx.state.get("final_last_attempt"):
                    final["refs"] = repair(ctx.agent_id, str(args.get("rationale", "")), args.get("refs"))
                    final["refs_repaired"] = True
                    return ToolResult("Answer submitted (invalid references dropped).", final=final)
                if err:
                    return ToolResult(err, ok=False)
                final["refs"] = refs
            return ToolResult("Answer submitted.", final=final)

        return ToolSpec(SUBMIT_NAME, self.submit_description(), {"type": "object", "properties": props,
                                                                 "required": req}, fn, terminal=True)

    def recorded_answer(self, answers: list[dict]) -> dict | None:
        """Answer an agent is credited with when it stops without submitting (last non-empty one)."""
        for a in reversed(answers or []):
            if a:
                return a
        return None

    def score_agent(self, example: dict, final: dict | None) -> float:
        raise NotImplementedError

    async def snapshot(self, example: dict, run) -> str:
        """The shared workspace as it stands (a patch or proof), for tasks that have one."""
        return ""

    def render_snapshot(self, state: str) -> str:
        return ""

    async def accepted(self, example: dict, final: dict | None) -> bool:
        """Whether the task's own checker accepts an answer during the run; False for tasks without one."""
        return False

    def score_group(self, example: dict, finals: list[dict], run=None) -> dict:
        """Pass@N: the group solves the example if any agent's final answer is correct."""
        per = []
        for f in finals:
            if f.get("failing"):
                per.append(None)
                continue
            per.append(float(self.score_agent(example, f.get("final"))) if f.get("final") else 0.0)
        valid = [p for p in per if p is not None]
        return {"score": max(valid) if valid else 0.0, "per_agent": per}
