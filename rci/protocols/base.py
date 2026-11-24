"""Protocols plug into the group runner through a small set of hooks."""
from __future__ import annotations

from rci.agents.tooling import ToolSpec


def is_failing(run, agent_id: int) -> bool:
    n, nf = run.cfg.n_agents, run.cfg.n_failing
    if getattr(run.cfg, "failing_position", "first") == "last":
        return agent_id >= n
    return agent_id < nf


class Protocol:
    """Base protocol: N members with the task's tools and no communication; subclasses override the hooks."""
    name = "base"
    note_cap: int | None = None

    def __init__(self, **kwargs):
        if kwargs:
            raise TypeError(f"{type(self).__name__} got unknown options {sorted(kwargs)}")

    def method_prompt(self, run, agent_id: int) -> str:
        return ""

    def build_system(self, run, agent_id: int, task_system: str) -> str | None:
        """Override to assemble the whole system prompt; None keeps method_prompt + task prompt."""
        return None

    def extra_tools(self, run, agent_id: int) -> list[ToolSpec]:
        return []

    def submit_tool(self, run, agent_id: int) -> ToolSpec:
        return run.task.submit_tool(run.example, with_refs=False)

    def build(self, run) -> None:
        """Default: N healthy agents plus `n_failing` agents that fail on every step."""
        n, nf = run.cfg.n_agents, run.cfg.n_failing
        for i in range(n + nf):
            system = run.task.system_prompt(run.example)
            mp = self.method_prompt(run, i)
            if mp:
                system = mp + "\n\n" + system
            custom = self.build_system(run, i, run.task.system_prompt(run.example))
            if custom is not None:
                system = custom
            tools = run.task.tools(run.example, run) + self.extra_tools(run, i) + [self.submit_tool(run, i)]
            run.add_agent(system, run.task.task_prompt(run.example, i, n + nf, run.rng), tools,
                          failing=is_failing(run, i))

    async def before_step(self, run, step: int) -> None:
        return None

    def injection(self, run, agent, step: int) -> str | None:
        """Appended to the agent's history as a user message before its turn (kept, summarised)."""
        return None

    def live_block(self, run, agent, step: int) -> str | None:
        """Re-rendered into the prompt every turn and never stored in the history."""
        return None

    def memory_blocks(self, run, agent) -> list[tuple[str, str]]:
        return []

    def on_prompt(self, run, agent, call: int, via: str) -> None:
        return None

    class Hold(str):
        """A gate text telling the member to wait: the refused call does not count as an action."""
        free = True

    def gate(self, run, agent, tool_name: str) -> str | None:
        return None

    def on_turn_end(self, run, agent, step: int) -> None:
        return None

    def on_agent_done(self, run, agent) -> None:
        return None

    async def after_step(self, run, step: int) -> None:
        return None

    def finished(self, run) -> bool:
        return all(a.done for a in run.agents)

    def needs_final(self, run, agent) -> bool:
        return True

    def defer_final(self, run, agent) -> bool:
        return False

    def collect(self, run) -> list[dict]:
        out = []
        for a in run.agents:
            out.append({"agent": a.id, "failing": a.failing, "role": a.role, "final": a.final,
                        "forced": a.final_forced})
        return out

    def dump(self, run) -> dict:
        return {}
