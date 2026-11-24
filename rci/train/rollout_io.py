"""Per-step rollout files and the worker that fills them."""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path

NEUTRAL_EXTRA = {"repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "min_p": 0.0}


def qfile(step_dir: Path, qid: str) -> Path:
    return step_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", qid) + ".jsonl")


def write_question(path: Path, qid: str, version: str, step: int, samples, summaries) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(json.dumps({"kind": "header", "question": qid, "version": version, "step": step,
                            "summaries": summaries}) + "\n")
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")
    os.replace(tmp, path)


def read_header(path: Path) -> dict | None:
    try:
        with open(path) as f:
            h = json.loads(f.readline())
        return h if h.get("kind") == "header" else None
    except (OSError, ValueError):
        return None


def read_question(path: Path):
    from rci.train.rollout import Sample

    with open(path) as f:
        header = json.loads(f.readline())
        samples = [Sample(**json.loads(line)) for line in f if line.strip()]
    return header, samples


def reusable(path: Path, allowed_version: str | None) -> bool:
    if allowed_version is None or not path.exists():
        return False
    h = read_header(path)
    return h is not None and h.get("version") == allowed_version


def run_question(job: dict) -> dict:
    """Subprocess entry point: K rollouts of one question -> one rollout file."""
    import asyncio
    import traceback

    from rci.agents.llm import LLMClient
    from rci.agents.runner import RunConfig
    from rci.registry import make_protocol, make_task
    from rci.train.rollout import RolloutConfig, rollout_question

    try:
        task = make_task(job["task"], **job["task_kwargs"])
        ex = job["example"]

        def client(urls, name):
            return LLMClient([u.rstrip("/") + "/v1" for u in urls], name, max_concurrency=job["llm_concurrency"],
                             default_extra=dict(NEUTRAL_EXTRA))

        policy = client(job["policy_urls"], job["served_policy"])
        peer = client(job["peer_urls"], job["served_peer"]) if job["peer_urls"] else policy
        summarize = task.summarize if job["summarize"] is None else job["summarize"]
        run_cfg = RunConfig(n_agents=job["n_agents"], max_actions=job["max_actions"], max_steps=job["max_steps"],
                            gen_limit=job["gen_limit"], temperature=1.0, top_p=1.0, top_k=-1,
                            enable_thinking=job["enable_thinking"], summarize=summarize, seed=job["seed"],
                            method_kwargs=job["protocol_kwargs"])
        rcfg = RolloutConfig(trainees=job["trainees"], credit=job["credit"], k=job["k"], beta=job["beta"], lam=job["lam"],
                             max_samples_per_agent=job["max_samples_per_agent"],
                             include_summaries=job["include_summaries"],
                             merge_prefix=job.get("merge_prefix", False), l_max=job.get("l_max", 32768),
                             adv_mode=job.get("adv_mode", "grpo"),
                             commit=job.get("commit", "top_confidence"), silent_floor=job.get("silent_floor", False),
                             seed=job["seed"])

        async def go():
            try:
                return await rollout_question(task, ex, lambda: make_protocol(job["protocol"], **job["protocol_kwargs"]),
                                              run_cfg, policy, peer, rcfg, job["step"])
            finally:
                for c in {id(x): x for x in policy.clients + peer.clients}.values():
                    await c.close()

        samples, summaries = asyncio.run(go())
        write_question(Path(job["path"]), ex["id"], job["version"], job["step"], samples, summaries)
        return {"question": ex["id"], "ok": True, "n_samples": len(samples)}
    except Exception as e:
        return {"question": job["example"].get("id"), "ok": False, "error": repr(e)[:500],
                "trace": traceback.format_exc()[-2000:]}
