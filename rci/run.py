"""Run one (task, protocol, group size) cell over examples and seeds, one resumable record per (seed, example)."""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from rci import paths
from rci.agents.capture import write_chains
from rci.agents.families import FAMILIES, family_name, family_of, record_info
from rci.agents.llm import LLMClient
from rci.agents.runner import GroupRun, RunConfig
from rci.registry import make_protocol, make_task

STOP = False
LOCK_STALE_S = 900
STOP_GRACE_S = 120


def _on_term(signum, frame):
    global STOP
    STOP = True
    print(f"[rci.run] received signal {signum}; stopping after in-flight runs are abandoned", flush=True)


def out_path(root: Path, protocol: str, n: int, seed: int, example_id: str, tag: str = "") -> Path:
    safe = str(example_id).replace("/", "_")
    return Path(root) / f"{protocol}{tag}" / f"N{n}" / f"s{seed}" / f"{safe}.json.gz"


def settings_mismatch(path: Path, protocol_kwargs: dict, task_kwargs: dict, model: str | None = None,
                      reasoning_effort: str | None = None, thinking: bool = True) -> str | None:
    """Why an existing record was made under other protocol or task settings than this run's, or None."""
    try:
        rec = json.load(gzip.open(path, "rt"))
    except Exception:
        return None
    mk, tk = (rec.get("config") or {}).get("method_kwargs"), rec.get("task_kwargs")
    if mk is not None and mk != protocol_kwargs:
        return f"protocol_kwargs {mk} != {protocol_kwargs}"
    if tk is not None and tk != task_kwargs:
        return f"task_kwargs {tk} != {task_kwargs}"
    if model is not None:
        rm = rec.get("model")
        if isinstance(rm, str) and "," not in rm and family_name(rm) != family_name(model):
            return f"model {rm} (family {family_name(rm)}) != {model} (family {family_name(model)})"
        mine = record_info(model, reasoning_effort, thinking)
        if "family" in rec or mine is not None:
            if rec.get("family") != mine:
                return f"family {rec.get('family')} != {mine}"
    return None


def try_lock(path: Path) -> bool:
    """Claim one example so concurrent drivers never run it twice; stale locks can be taken over."""
    owner_id = paths.job_id()
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, owner_id.encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            owner = lock.read_text().strip()
            age = time.time() - lock.stat().st_mtime
        except OSError:
            return False
        if owner == owner_id or age > LOCK_STALE_S:
            lock.write_text(owner_id)
            return True
        return False


def release_lock(path: Path) -> None:
    try:
        path.with_suffix(path.suffix + ".lock").unlink()
    except OSError:
        pass


async def heartbeat(held: set, every: float = 60) -> None:
    """Keep this driver's locks fresh until it is stopped."""
    while True:
        await asyncio.sleep(every)
        if STOP:
            return
        for p in list(held):
            try:
                os.utime(p.with_suffix(p.suffix + ".lock"))
            except OSError:
                pass


async def stop_watch(runs: asyncio.Future, grace: float | None = None, poll: float = 5) -> None:
    """Once stopped, give the runs in flight a grace period, then cancel them (cancelled runs are not written)."""
    while not STOP:
        await asyncio.sleep(poll)
    try:
        await asyncio.wait_for(asyncio.shield(runs), STOP_GRACE_S if grace is None else grace)
    except asyncio.TimeoutError:
        runs.cancel()
    except Exception:
        pass


def write_atomic(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _client(urls: str, name: str, concurrency: int, effort: dict) -> LLMClient:
    return LLMClient(urls.split(","), name, max_concurrency=concurrency,
                     **(effort if family_of(name).thinking == "effort" else {}))


async def main_async(args) -> int:
    task_kwargs = json.loads(args.task_kwargs)
    protocol_kwargs = json.loads(args.protocol_kwargs)
    protocol_name = make_protocol(args.protocol, **protocol_kwargs).name
    task = make_task(args.task, **dict(task_kwargs))
    examples = task.examples(limit=args.limit)
    if args.example_ids:
        keep = set(args.example_ids.split(","))
        examples = [e for e in examples if str(e["id"]) in keep]
    if args.shard is not None:
        i, m = map(int, args.shard.split("/"))
        examples = [e for j, e in enumerate(examples) if j % m == i]
    effort = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    llm = _client(args.base_urls, args.model, args.llm_concurrency, effort)
    if args.pool:
        pool = {name: _client(url, name, args.llm_concurrency, effort) for name, url in json.loads(args.pool).items()}
        if args.agent_models:
            names = args.agent_models.split(",")
            llm = {i: pool[n] for i, n in enumerate(names)}
            llm[-1] = pool[names[0]]
    root = Path(args.out)
    run_seeds = [int(s) for s in args.seeds.split(",")]
    pairs = ([(seed, ex) for ex in examples for seed in run_seeds] if args.order == "example"
             else [(seed, ex) for seed in run_seeds for ex in examples])
    todo, stale = [], []
    for seed, ex in pairs:
        p = out_path(root, protocol_name, args.n_agents, seed, ex["id"], args.tag)
        if p.exists() and not args.overwrite:
            why = settings_mismatch(p, protocol_kwargs, task_kwargs, model=None if args.pool else args.model,
                                    reasoning_effort=args.reasoning_effort, thinking=not args.no_thinking)
            if why:
                stale.append(f"{p}: {why}")
            continue
        todo.append((seed, ex, p))
    if stale:
        raise SystemExit(f"[rci.run] {len(stale)} existing records were made under other settings; use another --out or "
                         f"--overwrite. First: {stale[0]}")
    print(f"[rci.run] {args.task}/{protocol_name} N={args.n_agents}: {len(todo)} runs to do "
          f"({len(examples)} examples)", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    t_start = time.time()
    done = {"n": 0, "tokens": 0, "score": 0.0, "failed": 0}
    held: set = set()
    hb = asyncio.create_task(heartbeat(held))

    async def one(seed, ex, p):
        if STOP:
            return
        async with sem:
            if STOP or p.exists() or not try_lock(p):
                return
            held.add(p)
            try:
                await run_one(seed, ex, p)
            finally:
                held.discard(p)
                release_lock(p)

    async def run_one(seed, ex, p):
        cfg = RunConfig(
            n_agents=args.n_agents, max_actions=args.max_actions or task.max_actions,
            max_steps=args.max_steps or task.max_steps, gen_limit=args.gen_limit,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, presence_penalty=args.presence_penalty,
            enable_thinking=not args.no_thinking, summarize=task.summarize and not args.no_summary, seed=seed,
            save_transcripts=args.save_transcripts, method_kwargs=protocol_kwargs)
        protocol = make_protocol(args.protocol, **protocol_kwargs)
        try:
            run = GroupRun(task, ex, protocol, cfg, llm)
            run.capture_agents = "all" if args.capture else set()
            rec = await run.run()
            if args.capture:
                write_chains(p, run.agents)
        except Exception as e:
            print(f"[rci.run] FAILED {ex['id']} s{seed}: {e!r}\n{traceback.format_exc()}", flush=True)
            done["failed"] += 1
            return
        rec["model"] = args.model if not args.pool else args.agent_models
        rec["task_kwargs"] = task_kwargs
        fam = None if args.pool else record_info(args.model, args.reasoning_effort, thinking=not args.no_thinking)
        if fam is not None:
            rec["family"] = fam
        write_atomic(p, rec)
        done["n"] += 1
        done["tokens"] += rec["compute"]["total_tokens"]
        sc = rec["score"].get("score")
        done["score"] += float(sc) if sc is not None else 0.0
        el = time.time() - t_start
        print(f"[rci.run] done {done['n']}/{len(todo)} {ex['id']} s{seed} score={sc if sc is not None else 'pending'} "
              f"tokens={rec['compute']['total_tokens']} wall={rec['wall_time']}s | mean score "
              f"{done['score'] / done['n']:.3f}, {done['tokens'] / max(el, 1):.0f} tok/s, "
              f"{done['n'] / max(el, 1) * 3600:.1f} runs/h", flush=True)

    runs = asyncio.gather(*(one(*t) for t in todo))
    watch = asyncio.create_task(stop_watch(runs))
    try:
        await runs
    except asyncio.CancelledError:
        print(f"[rci.run] in-flight runs cancelled {STOP_GRACE_S}s after the stop signal", flush=True)
    watch.cancel()
    hb.cancel()
    if done["failed"]:
        print(f"[rci.run] {done['failed']} runs failed and were not written; a later launch retries them", flush=True)
    return 143 if STOP else 0


SAMPLING = {"qwen_thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                              "gen_limit": 32768}}
SAMPLING.update({n: dict(f.sampling) for n, f in FAMILIES.items() if n != "qwen"})


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rci.run",
                                 description="Run one (task, protocol, group size) cell and write one record per "
                                             "(seed, example).")
    ap.add_argument("--task", required=True)
    ap.add_argument("--task-kwargs", default="{}", help="JSON object of task options")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--protocol-kwargs", default="{}", help="JSON object of protocol options")
    ap.add_argument("--n-agents", type=int, default=1)
    ap.add_argument("--seeds", default="0", help="comma list of seeds")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--example-ids", default=None, help="comma list of example ids to run")
    ap.add_argument("--shard", default=None, help="i/m: run every m-th example starting at i")
    ap.add_argument("--base-urls", required=True, help="comma list of OpenAI-compatible endpoints")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--pool", default=None, help="JSON {model_name: base_url[,url2]} for groups of several models")
    ap.add_argument("--agent-models", default=None, help="comma list, model name per member (with --pool)")
    ap.add_argument("--out", required=True, help="root directory of the records")
    ap.add_argument("--tag", default="", help="suffix of the protocol directory")
    ap.add_argument("--concurrency", type=int, default=4, help="group runs in flight")
    ap.add_argument("--llm-concurrency", type=int, default=256)
    ap.add_argument("--max-actions", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--gen-limit", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--sampling", default=None, choices=sorted(SAMPLING),
                    help="a recommended sampling preset (temperature, top-p, top-k, presence penalty; qwen_thinking "
                         "also sets the generation limit)")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"],
                    help="reasoning effort of thinking calls for an effort family (gpt-oss)")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--save-transcripts", action="store_true")
    ap.add_argument("--capture", action="store_true", help="also store every member's token ids next to the record")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--order", choices=("seed", "example"), default="seed",
                    help="run order: seed-major (default) or example-major")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.reasoning_effort and family_of(args.model).thinking != "effort" and not args.pool:
        ap.error(f"--reasoning-effort applies to an effort family (gpt-oss); {args.model} is {family_of(args.model).name}")
    for k, v in SAMPLING.get(args.sampling, {}).items():
        setattr(args, k, v)
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
