"""Tool specifications shared by tasks and methods."""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class ToolResult:
    text: str
    ok: bool = True
    final: dict | None = None
    content: list[dict] | None = None
    free: bool = False
    meta: dict | None = None


ToolFn = Callable[["AgentCtx", dict], "ToolResult | str | Awaitable[ToolResult | str]"]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: ToolFn
    terminal: bool = False
    free: bool = False
    per_turn: int = 0
    on_compact: str = "keep"

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class AgentCtx:
    """What a tool sees when it runs: the calling agent and the run it belongs to."""
    agent_id: int
    run: Any
    step: int = 0
    state: dict = field(default_factory=dict)


async def call_tool(spec: ToolSpec, ctx: AgentCtx, raw_args: str | dict) -> ToolResult:
    if isinstance(raw_args, dict):
        args = raw_args
    else:
        try:
            args = json.loads(raw_args) if raw_args and raw_args.strip() else {}
        except json.JSONDecodeError as e:
            return ToolResult(f"Error: arguments to {spec.name} are not valid JSON ({e}).", ok=False)
    if not isinstance(args, dict):
        return ToolResult(f"Error: arguments to {spec.name} must be a JSON object.", ok=False)
    try:
        out = spec.fn(ctx, args)
        if inspect.isawaitable(out):
            out = await out
    except (OSError, TimeoutError, ConnectionError, RuntimeError):
        raise
    except Exception as e:
        return ToolResult(f"Error: bad arguments to {spec.name}: {type(e).__name__}: {e}", ok=False)
    if isinstance(out, str):
        return ToolResult(out)
    return out


def obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or list(properties)}


STR = {"type": "string"}
STR_LIST = {"type": "array", "items": {"type": "string"}}
INT = {"type": "integer"}
