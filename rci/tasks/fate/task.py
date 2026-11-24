"""FATE: formal abstract algebra theorems in Lean 4 + Mathlib."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rci import paths
from rci.agents.tooling import STR, AgentCtx, ToolResult, ToolSpec
from rci.tasks.base import Task

SYSTEM = """You are an expert in Lean 4 and Mathlib. Prove the given theorem.

Tools:
- search_mathlib(query): the 10 best-matching Mathlib declarations (names, signatures, docstrings).
- check_lean(code): compiles a complete Lean file and returns the compiler messages. You may call it at
  most {max_checks} times, so make each check count.

Your final proof must be a complete Lean file that starts with `import Mathlib`, keeps the theorem name
and statement exactly as given (outside any namespace), contains no `sorry`, and compiles without
errors. Submit it with `submit_answer`."""

MAX_CHECKS, CHECK_TIMEOUT = 10, 900.0
SNAPSHOT_CHARS = 8_000


class FATE(Task):
    name = "fate"
    msg_token_limit = 1024
    max_actions = 100
    max_steps = 120
    summarize = True

    def __init__(self, split: str = "M", root: str = "", service: str = "",
                 limit_default: int = 100, file: str | None = None):
        self.split, self.limit_default = split.upper(), limit_default
        self.root = Path(root) if root else paths.data_dir("fate")
        self.service = service or str(paths.data_dir("fate", "service", "current.json"))
        self.file = file
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from rci.tasks.fate.lean_service.client import FAILOVER_WAIT_S, LeanClient
            self._client = LeanClient.from_addr_file(self.service, wait_s=FAILOVER_WAIT_S)
        return self._client

    def examples(self, limit: int | None = None) -> list[dict]:
        if self.file:
            rows = [json.loads(l) for l in open(self.root / self.file) if l.strip()]
            rows = sorted(rows, key=lambda r: r["idx"])[:limit] if limit else sorted(rows, key=lambda r: r["idx"])
            return [{"id": r["pid"].replace("/", "_"), **r} for r in rows]
        f = self.root / ("fate_m.jsonl" if self.split == "M" else "fate_h.jsonl")
        rows = [json.loads(l) for l in open(f) if l.strip()]
        rows = [r for r in rows if r.get("eval", True)][: (limit or self.limit_default)]
        return [{"id": r["pid"].replace("/", "_"), **r} for r in rows]

    def _check(self, code: str, example: dict) -> dict:
        if example.get("adhoc"):
            return self.client.check(code, None, example["theorem"], example["formal_statement"], CHECK_TIMEOUT)
        return self.client.check(code, example["pid"], None, None, CHECK_TIMEOUT)

    def system_prompt(self, example: dict) -> str:
        return SYSTEM.format(max_checks=MAX_CHECKS)

    def task_prompt(self, example: dict, agent_id: int, n_agents: int, rng) -> str:
        return ("Prove the following theorem. Replace `sorry` with a complete proof; do not change the "
                f"statement.\n\n```lean\n{example['formal_statement']}\n```")

    def tools(self, example: dict, run) -> list[ToolSpec]:
        seen = run.state.setdefault("fate_seen_decls", set()) if run is not None else set()

        async def search(ctx: AgentCtx, args: dict) -> ToolResult:
            q = str(args.get("query", "")).strip()
            if not q:
                return ToolResult("Error: empty query.", ok=False)
            r = await asyncio.to_thread(self.client.search, q, 10)
            for d in r.get("results", []):
                seen.add(d["name"])
            return ToolResult(r.get("text", "No matching Mathlib declarations."))

        async def check(ctx: AgentCtx, args: dict) -> ToolResult:
            used = ctx.state.get("lean_checks", 0)
            if used >= MAX_CHECKS:
                return ToolResult(f"Error: you have used all {MAX_CHECKS} Lean checks.", ok=False)
            code = str(args.get("code", ""))
            if not code.strip():
                return ToolResult("Error: empty code.", ok=False)
            ctx.state["lean_checks"] = used + 1
            r = await asyncio.to_thread(self._check, code, example)
            left = MAX_CHECKS - used - 1
            return ToolResult(f"{r.get('messages', r.get('status', ''))}\n\n({left} Lean checks left)",
                              ok=r.get("accepted", False) or r.get("status") != "error")

        return [
            ToolSpec("search_mathlib", "Return the 10 best-matching Mathlib declarations for a query.",
                     {"type": "object", "properties": {"query": {**STR, "description": "Name fragment or description"}},
                      "required": ["query"]}, search),
            ToolSpec("check_lean", f"Compile a complete Lean 4 file (at most {MAX_CHECKS} calls per agent).",
                     {"type": "object", "properties": {"code": {**STR, "description": "Complete Lean file"}},
                      "required": ["code"]}, check),
        ]

    def resource_exists(self, example: dict, run, ref: str) -> bool:
        name = ref[len("Mathlib:"):] if ref.startswith("Mathlib:") else ref
        return run is not None and name in run.state.get("fate_seen_decls", set())

    async def snapshot(self, example: dict, run) -> str:
        """What the last seat left: its proof and what the checker says about it."""
        last = [a for a in run.agents if a.done][-1]
        proof = str((last.final or {}).get("proof") or "")
        if not proof.strip():
            return json.dumps({"proof": "", "messages": "", "accepted": False})
        r = await asyncio.to_thread(self._check, proof, example)
        return json.dumps({"proof": proof, "messages": str(r.get("messages", r.get("status", ""))),
                           "accepted": bool(r.get("accepted"))})

    def render_snapshot(self, state: str) -> str:
        s = json.loads(state) if state else {"proof": ""}
        if not s["proof"].strip():
            return "They left no proof."
        return (f"Their last proof:\n\n```lean\n{s['proof'][:SNAPSHOT_CHARS]}\n```\n\n"
                f"Lean says:\n\n{s['messages'][:SNAPSHOT_CHARS]}")

    def answer_from_attachment(self, text: str) -> dict:
        """A DeLM attachment on FATE is a Lean file; it is the agent's recorded candidate proof."""
        return {"proof": text}

    def answer_properties(self):
        return {"proof": {**STR, "description": "The complete Lean file proving the theorem."}}, ["proof"]

    def score_agent(self, example: dict, final: dict | None) -> float:
        if not final or not str(final.get("proof", "")).strip():
            return 0.0
        key = (example["pid"], hash(str(final["proof"])))
        cache = self.__dict__.setdefault("_score_cache", {})
        if key not in cache:
            r = self._check(str(final["proof"]), example)
            cache[key] = 1.0 if r.get("accepted") else 0.0
        return cache[key]

    async def accepted(self, example: dict, final: dict | None) -> bool:
        return await asyncio.to_thread(self.score_agent, example, final) > 0.5

    async def score_group(self, example: dict, finals: list[dict], run=None) -> dict:
        async def one(f):
            if f.get("failing"):
                return None
            return await asyncio.to_thread(self.score_agent, example, f.get("final"))
        per = await asyncio.gather(*(one(f) for f in finals))
        valid = [p for p in per if p is not None]
        return {"score": max(valid) if valid else 0.0, "per_agent": list(per)}
