"""Write SplitBench candidates with a served model; keep those with the right shape and far from HiddenBench."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

from rci import paths
from rci.agents.llm import LLMClient
from rci.tasks.splitbench.writer import novelty, parse, prompt, to_benchmark


async def write_all(llm: LLMClient, n: int, seed: int, concurrency: int, raw_path: Path, control: bool = False,
                    n_hidden: int | None = None) -> list[dict]:
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            r = json.loads(line)
            done[r["i"]] = r
    sem = asyncio.Semaphore(concurrency)
    f = open(raw_path, "a")

    async def one(i: int):
        if i in done:
            return
        text, draw = prompt(random.Random(f"{seed}:{i}"), control=control, n_hidden=n_hidden)
        async with sem:
            try:
                call = await llm.chat([{"role": "user", "content": text}], max_tokens=12000, temperature=1.0,
                                      enable_thinking=True, tag="splitbench", route_key=f"g{i}")
                reply = call.content or ""
            except Exception as e:
                print(f"[splitbench] {i} failed: {e!r}", flush=True)
                return
        r = {"i": i, "draw": draw, "reply": reply}
        done[i] = r
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        if len(done) % 50 == 0:
            print(f"[splitbench] {len(done)}/{n}", flush=True)

    await asyncio.gather(*(one(i) for i in range(n)))
    f.close()
    return [done[i] for i in sorted(done)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.splitbench.generate",
                                 description="Write SplitBench candidate tasks with a served model.")
    ap.add_argument("--base-url", required=True, help="comma list of OpenAI-compatible endpoints")
    ap.add_argument("--model", required=True, help="served model name of the writer")
    ap.add_argument("--n", type=int, default=1500, help="number of writing prompts")
    ap.add_argument("--out", default=None, help="default: $RCI_DATA/splitbench/raw (or splitbench_control/raw)")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-novelty", type=float, default=0.30, help="largest word-set Jaccard against HiddenBench")
    ap.add_argument("--bench", default=None, help="HiddenBench benchmark.json (default: $RCI_DATA/hiddenbench)")
    ap.add_argument("--control", action="store_true", help="the reverse case: shared facts point to the right option")
    ap.add_argument("--start", type=int, default=None, help="first numeric id (default 1000; 3000 with --control)")
    ap.add_argument("--n-hidden", type=int, default=None, help="fix the number of private facts")
    a = ap.parse_args(argv)
    a.out = a.out or str(paths.data_dir("splitbench_control" if a.control else "splitbench", "raw"))
    a.bench = a.bench or str(paths.data_dir("hiddenbench", "benchmark.json"))
    a.start = a.start if a.start is not None else (3000 if a.control else 1000)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLMClient(a.base_url.split(","), a.model, max_concurrency=a.concurrency)
    raw = asyncio.run(write_all(llm, a.n, a.seed, a.concurrency, out / "raw.jsonl", a.control, a.n_hidden))
    bench = json.loads(Path(a.bench).read_text())
    kept, why = [], Counter()
    for r in raw:
        t = parse(r["reply"], r["draw"])
        if t is None:
            why["shape"] += 1
            continue
        nv = novelty(t, bench)
        if nv > a.max_novelty:
            why["near a benchmark task"] += 1
            continue
        kept.append({**t, "novelty": round(nv, 3), "draw_i": r["i"]})
    rows = to_benchmark(kept, start=a.start)
    (out / "benchmark.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    report = {"written": len(raw), "kept": len(rows), "dropped": dict(why),
              "right_position": dict(Counter(r["possible_answers"].index(r["correct_answer"]) for r in rows)),
              "n_hidden": dict(Counter(len(r["hidden_information"]) for r in rows)),
              "max_novelty": max((r["novelty"] for r in rows), default=None)}
    (out / "REPORT.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
