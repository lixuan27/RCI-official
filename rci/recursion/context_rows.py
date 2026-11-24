"""Rows with the free-debate prompt as context: verified first turns moved into it, or sampled in it."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from rci.recursion import resample
from rci.recursion.rows import row_of, scan, write_rows
from rci.recursion.turns import guard_report, load_bench, no_truth, protocol_dirs, seeds_of

MAX_LEN = 15427


def transplant(a) -> dict:
    """Verified forwarding-debate first turns, each placed after the free-debate prompt of the same seat."""
    dirs = protocol_dirs(a.protocol_dirs)
    stats: dict = {}
    with no_truth():
        bench, sets = load_bench(a.data.split(","))
        runs = a.harvest.split(",")
        turns = scan(runs, bench, dirs, ("a",), seeds_of(a.seeds), "keep", None, set(), stats)
        f_seeds = seeds_of(a.f_seed) if a.f_seed else seeds_of(a.seeds)
        f_ctx = {}
        for c in sorted(resample.contexts(runs, bench, dirs, ("f",), f_seeds, Counter()), key=lambda c: c["seed"]):
            f_ctx.setdefault((c["qid"], c["seat"]), c["prefix"])
        rows, missing, cache = [], 0, {}
        for t in sorted((t for t in turns if t["ok"]), key=lambda t: (t["qid"], t["key"])):
            pre = f_ctx.get((t["qid"], t["seat"]))
            if pre is None:
                missing += 1
                continue
            r = row_of(t, cache)
            start, n = r["spans"][0]
            gen = r["ids"][start:start + n]
            if len(pre) + n > a.max_len:
                continue
            rows.append({"qid": t["qid"], "protocol": "f", "seed": t["seed"], "seat": t["seat"], "w": 1.0,
                         "ids": pre + gen, "spans": [[len(pre), n]]})
    if missing and not a.allow_missing:
        raise SystemExit(f"{missing} verified turns have no free-debate prompt of the same question and seat")
    out = Path(a.out)
    digest = write_rows(out, rows)
    rep = {"label_free": True, "truth_guard": guard_report(), "sets": sets, "verified": sum(t["ok"] for t in turns),
           "rows": len(rows), "without_context": missing,
           "reasons": {k: dict(v) for k, v in stats.items()}, "sha256": digest}
    out.with_suffix(".report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("verified", "rows", "sha256")}))
    return rep


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rows in the free-debate context. Without --base-urls: the harvest's "
                                             "verified first turns placed after the free-debate prompt of the same "
                                             "seat. With --base-urls: K first turns sampled in the free-debate "
                                             "prompts, the shortest verified one kept, its target ended before any "
                                             "tool call.")
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--protocol-dirs", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--f-seed", default=None, help="harvest seed of the free-debate prompts (default: the lowest)")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--base-urls", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tokenizer", default=None, help="model directory; needed when sampling")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not a.base_urls:
        transplant(a)
        return 0
    if not a.model or not a.tokenizer:
        ap.error("sampling needs --model and --tokenizer")
    ns = argparse.Namespace(harvest=a.harvest, data=a.data, base_urls=a.base_urls, model=a.model, k=a.k,
                            protocols="f", protocol_dirs=a.protocol_dirs, seeds=a.f_seed or a.seeds,
                            criterion="loose", target="cut", tokenizer=a.tokenizer, pick="shortest",
                            transplant_f=False, max_tokens=a.max_tokens, max_len=a.max_len,
                            concurrency=a.concurrency, out=a.out)
    asyncio.run(resample.run(ns))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
