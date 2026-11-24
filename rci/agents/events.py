"""The ordered run-wide event log a task can keep in each member's task state."""
from __future__ import annotations


def enabled(run) -> bool:
    """Whether the run keeps the event log (run.state["event_log"] is True)."""
    state = getattr(run, "state", None)
    return isinstance(state, dict) and state.get("event_log") is True


def next_seq(run) -> int:
    """The next run-wide sequence number (increments run.state["event_seq"] and returns the new value)."""
    run.state["event_seq"] = int(run.state.get("event_seq", 0)) + 1
    return run.state["event_seq"]


def log(run, agent_id: int, op: str, **fields) -> dict | None:
    if not enabled(run):
        return None
    ev = {"seq": next_seq(run), "op": op, "step": getattr(run, "step", None), "agent": agent_id, **fields}
    run.agents[agent_id].ctx.state.setdefault("events", []).append(ev)
    return ev
