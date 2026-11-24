"""SWE-bench Verified: one bash tool in a rootless sandbox; the diff left in the workspace is the answer."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from rci import paths
from rci.agents.tooling import STR, AgentCtx, ToolResult, ToolSpec
from rci.tasks.base import Task

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCH_TRIES, LAUNCH_WAIT_S = 3, 10

OUTPUT_CHARS = 10_000
SNAPSHOT_CHARS = 20_000
PATCH_KEEP_CHARS = 200_000
CANDIDATE_CHUNK = 60_000
ENV = {"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1",
       "GIT_PAGER": "cat", "PYTHONDONTWRITEBYTECODE": "1"}
PRELUDE = ("[ -f /opt/miniconda3/bin/activate ] && . /opt/miniconda3/bin/activate testbed >/dev/null 2>&1; "
           "cd /testbed || exit 1; ")
DIFF_CMD = "git add -A >/dev/null 2>&1 && git diff --cached --no-color"

SYSTEM = """You are a software engineer resolving an issue in a Python repository checked out at /testbed.

Tool:
- bash(command): runs one command with bash in /testbed and returns its output (stdout and stderr). Every call
  starts a new shell: `cd` and environment variables do not carry over, but changes to files do. Commands time
  out after {timeout} seconds. There is no interactive terminal: edit files with sed, python or
  `cat <<'EOF' > file`, and never start editors or pagers.

Work method: locate the relevant code, write a short script that reproduces the issue, change the source code
(not the tests) to fix it, re-run your script and a few edge cases, then call submit_answer with a short summary.
The state of /testbed when you stop (its git diff) is your answer, so leave only the changes that fix the issue."""


def _truncate(out: str) -> str:
    if len(out) <= OUTPUT_CHARS:
        return out
    h = OUTPUT_CHARS // 2
    return (out[:h] + f"\n[... {len(out) - OUTPUT_CHARS} characters omitted; narrow your command ...]\n"
            + out[-h:])


def _iid(example: dict) -> str:
    return example.get("instance_id") or example["id"]


class SWEBenchVerified(Task):
    name = "swe"
    msg_token_limit = 1024
    max_actions = 300
    max_steps = 400
    summarize = False

    def __init__(self, parquet: str = "", store: str = "", cache: str = "", first: int = 100, verify: str | None = "",
                 cmd_timeout: int = 180, eval_timeout: int = 1800, eval_concurrency: int = 4,
                 eval_python: str = "", cache_budget_gb: float | None = None,
                 executor=None, evaluator=None, remover=None):
        parquet = parquet or str(paths.data_dir("swe", "hf", "verified.parquet"))
        store = store or str(paths.data_dir("swe", "oci"))
        cache = cache or str(paths.cache_dir("swe"))
        if verify == "":
            verify = str(paths.data_dir("swe", "oci", "verify_first100.json"))
        self.parquet, self.store, self.cache = parquet, os.path.abspath(store), cache
        self.cache_budget = cache_budget_gb * 1e9 if cache_budget_gb else None
        self.first, self.verify = first, verify
        self.cmd_timeout, self.eval_timeout = cmd_timeout, eval_timeout
        self.eval_concurrency = eval_concurrency
        self._sems: dict[int, asyncio.Semaphore] = {}
        self.eval_python = eval_python or paths.env("RCI_SWE_PYTHON") or sys.executable
        self._exec = executor or self._sandbox_exec
        self._evaluate = evaluator or self._sandbox_evaluate
        self._remove = remover

    def examples(self, limit: int | None = None) -> list[dict]:
        import pyarrow.parquet as pq

        rows = pq.read_table(self.parquet).to_pylist()[: self.first]
        ok = None
        if self.verify and os.path.exists(self.verify):
            rep = json.loads(Path(self.verify).read_text())
            ok = {k for k, v in rep.items() if not v.get("problems")}
        out = []
        for r in rows:
            if ok is not None and r["instance_id"] not in ok:
                continue
            out.append({"id": r["instance_id"], "instance_id": r["instance_id"], "repo": r["repo"],
                        "base_commit": r["base_commit"], "problem_statement": r["problem_statement"]})
        return out[:limit] if limit else out

    def system_prompt(self, example: dict) -> str:
        return SYSTEM.format(timeout=self.cmd_timeout)

    def task_prompt(self, example: dict, agent_id: int, n_agents: int, rng) -> str:
        return (f"Repository {example['repo']} at commit {example['base_commit'][:12]}.\n\n<issue>\n"
                f"{example['problem_statement'].strip()}\n</issue>\n\nResolve the issue.")

    def _rundir(self, run, agent_id: int) -> str:
        dirs = run.state.setdefault("swe_dirs", {})
        if run.state.get("shared_workspace"):
            agent_id = 0
        if agent_id not in dirs:
            key = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{run.run_id}_{os.getpid()}_{id(run)}")[-150:]
            dirs[agent_id] = os.path.join(self.cache, "runs", "agents", key, f"a{agent_id}")
        return dirs[agent_id]

    async def _pinned(self, example: dict, run):
        fut = run.state.get("swe_pin")
        if fut is None:
            from rci.tasks.swe import sandbox as swe_sandbox

            def pin_and_evict() -> str:
                p = swe_sandbox.pin(self.store, self.cache, _iid(example), str(id(run)))
                swe_sandbox.evict(self.store, self.cache, self.cache_budget)
                return p

            fut = run.state["swe_pin"] = asyncio.ensure_future(asyncio.to_thread(pin_and_evict))
        return await fut

    def _sandbox_exec(self, example: dict, rundir: str, shell_cmd: str, timeout: float) -> tuple[int, str]:
        from rci.tasks.swe import sandbox as swe_sandbox

        return swe_sandbox.exec_cmd(self.store, _iid(example), self.cache, rundir, PRELUDE + shell_cmd,
                                    timeout, extra_env=ENV)

    def tools(self, example: dict, run) -> list[ToolSpec]:
        async def bash(ctx: AgentCtx, args: dict) -> ToolResult:
            cmd = str(args.get("command", "")).strip()
            if not cmd:
                return ToolResult("Error: empty command.", ok=False)
            rundir = self._rundir(ctx.run, ctx.agent_id)
            if self.cache_budget:
                await self._pinned(example, ctx.run)
            rc, out = await asyncio.to_thread(self._exec, example, rundir, cmd, self.cmd_timeout)
            if rc == 124:
                return ToolResult(_truncate(out) + f"\n[command timed out after {self.cmd_timeout} s]", ok=False)
            return ToolResult(f"[exit code {rc}]\n" + _truncate(out))

        return [ToolSpec("bash", "Run one bash command in /testbed (fresh shell; file changes persist).",
                         {"type": "object", "properties": {"command": {**STR, "description": "The command"}},
                          "required": ["command"]}, bash)]

    def answer_properties(self):
        return {"summary": {**STR, "description": "What you changed and how you checked it."}}, ["summary"]

    def submit_description(self) -> str:
        return "Finish. The current changes in /testbed (its git diff) are your answer."

    def recorded_answer(self, answers: list[dict]) -> dict | None:
        return {"summary": "(recorded at stop)"}

    def answer_from_attachment(self, text: str) -> dict:
        return {"summary": text[:2000]}

    async def snapshot(self, example: dict, run) -> str:
        """The shared workspace as it stands: its diff."""
        d = (run.state.get("swe_dirs") or {}).get(0)
        return (await asyncio.to_thread(self._diff, example, d)) if d else ""

    def render_snapshot(self, state: str) -> str:
        if not state.strip():
            return "/testbed has no changes yet."
        body = state if len(state) <= SNAPSHOT_CHARS else state[:SNAPSHOT_CHARS] + "\n[... diff truncated; run `git diff` ...]"
        return f"/testbed holds their changes (`git diff`):\n\n```diff\n{body}\n```"

    def stage_candidates(self, example: dict, run, patches: dict[str, str]) -> None:
        """Puts each candidate at /opt/review/candidate_<label>.diff of the reviewer's workspace (outside /testbed)."""
        rundir = self._rundir(run, 0)
        for label, patch in patches.items():
            enc = base64.b64encode(patch.encode()).decode()
            self._exec(example, rundir, f"mkdir -p /opt/review && : > /opt/review/{label}.b64", 60)
            for i in range(0, len(enc), CANDIDATE_CHUNK):
                self._exec(example, rundir, f"printf %s {enc[i:i + CANDIDATE_CHUNK]} >> /opt/review/{label}.b64", 60)
            rc, out = self._exec(example, rundir, f"base64 -d /opt/review/{label}.b64 > /opt/review/candidate_{label}.diff", 60)
            if rc != 0:
                raise RuntimeError(f"could not stage candidate {label}: {out[:200]}")

    def release(self, run) -> None:
        if self._remove is None:
            from rci.tasks.swe import sandbox as swe_sandbox

            self._remove = swe_sandbox.rmtree_ns
        for d in (run.state.get("swe_dirs") or {}).values():
            self._remove(d)

    def _diff(self, example: dict, rundir: str) -> str:
        if not os.path.isdir(os.path.join(rundir, "upper")):
            return ""
        rc, out = self._exec(example, rundir, DIFF_CMD, 300)
        return out if rc == 0 else ""

    def _sandbox_evaluate(self, example: dict, patch: str) -> dict:
        import subprocess

        with tempfile.TemporaryDirectory(dir=self.cache if os.path.isdir(self.cache) else None) as d:
            pf = os.path.join(d, "patch.diff")
            Path(pf).write_text(patch)
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))
            argv = [self.eval_python, "-m", "rci.tasks.swe.evaluate", "--instance", _iid(example),
                    "--patch-file", pf, "--out", d, "--store", self.store, "--cache", self.cache,
                    "--parquet", os.path.abspath(self.parquet), "--timeout", str(self.eval_timeout)]
            for attempt in range(LAUNCH_TRIES):
                try:
                    subprocess.run(argv, timeout=self.eval_timeout + 900, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, check=False, env=env)
                    break
                except subprocess.TimeoutExpired:
                    return {"resolved": False, "error": "evaluation timed out"}
                except OSError:
                    if attempt == LAUNCH_TRIES - 1:
                        raise
                    time.sleep(LAUNCH_WAIT_S)
            rp = os.path.join(d, "report.json")
            if not os.path.exists(rp):
                raise RuntimeError(f"SWE evaluation produced no report for {example['id']}")
            return json.loads(Path(rp).read_text())

    async def score_group(self, example: dict, finals: list[dict], run=None) -> dict:
        dirs = dict((run.state.get("swe_dirs") or {}) if run is not None else {})
        patches: dict[int, str] = {}
        for f in finals:
            if f.get("failing"):
                continue
            d = dirs.get(0 if run is not None and run.state.get("shared_workspace") else f["agent"])
            if d is None and f.get("role") == "coordinator" and dirs:
                d = dirs[max(dirs)]
            patches[f["agent"]] = (await asyncio.to_thread(self._diff, example, d)) if d else ""
        stage = dict((run.state.get("stage") or {}) if run is not None else {})
        uniq = sorted({p for p in [*patches.values(), *stage.values()] if p.strip()})

        loop = id(asyncio.get_running_loop())
        sem = self._sems.setdefault(loop, asyncio.Semaphore(self.eval_concurrency))

        async def ev(p: str) -> dict:
            async with sem:
                return await asyncio.to_thread(self._evaluate, example, p)

        reports = dict(zip(uniq, await asyncio.gather(*(ev(p) for p in uniq))))
        per = []
        for f in finals:
            if f.get("failing"):
                per.append(None)
                continue
            p = patches.get(f["agent"], "")
            per.append(1.0 if p.strip() and reports[p].get("resolved") else 0.0)
        if self._remove is None:
            from rci.tasks.swe import sandbox as swe_sandbox

            self._remove = swe_sandbox.rmtree_ns
        for d in dirs.values():
            await asyncio.to_thread(self._remove, d)
        if run is not None and run.state.get("swe_pin") is not None:
            from rci.tasks.swe import sandbox as swe_sandbox

            swe_sandbox.unpin(await run.state.pop("swe_pin"))
        valid = [x for x in per if x is not None]
        out = {"score": max(valid) if valid else 0.0, "per_agent": per, "n_patches": len(uniq),
               "patch_chars": {str(a): len(p) for a, p in patches.items()},
               "patches": {str(a): p[:PATCH_KEEP_CHARS] for a, p in patches.items()},
               "eval": [{k: v for k, v in r.items() if k in ("resolved", "error", "tests_status_counts", "timing")}
                        for r in reports.values()]}
        if stage:
            out["stage_resolved"] = {str(a): bool(p.strip() and reports[p].get("resolved")) for a, p in sorted(stage.items())}
        return out
