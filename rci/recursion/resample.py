"""Second-generation and control rows: K first turns per seat, the shortest verified one kept."""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

from rci.agents.capture import chains_path, read_chains
from rci.recursion.rows import write_rows
from rci.recursion.turns import (guard_report, harvest_records, has_tool_call, ids_hash, load_bench, mine_of, no_truth,
                                 protocol_dirs, seeds_of, sha, special_ids)
from rci.recursion.verify import CRITERIA, first_span_row, judge

PICKS = ("shortest", "random")
TARGETS = ("whole", "cut")


def contexts(runs: list, bench: dict, dirs: dict, protocols: tuple, seeds: set | None, stats: Counter) -> list:
    """(context id, question, protocol, seat, private facts, prompt ids) of every member with a captured first span."""
    out = []
    for proto in protocols:
        for p, seed, rec in harvest_records(runs, dirs[proto], 4, seeds):
            q = str(rec["example_id"])
            row = bench.get(q)
            if row is None:
                continue
            if not chains_path(Path(p)).exists():
                stats["records without chains"] += 1
                continue
            chains, n = read_chains(Path(p)), len(rec["agents"])
            for j in range(n):
                mine = mine_of(row, n, j)
                if not mine:
                    continue
                r = first_span_row(chains.get(j, []))
                if r is None:
                    stats["members without a captured first span"] += 1
                    continue
                out.append({"cid": f"{proto}/{q}/s{seed}/a{j}", "qid": q, "proto": proto, "seed": seed, "seat": j,
                            "mine": mine, "prefix": r["ids"][:r["spans"][0][0]]})
    return out


async def sample(clients, router, model: str, key: str, prefix: list, max_tokens: int, sem):
    """One completion from the prompt ids: (text, generated ids, finish reason), or None when the call failed."""
    async with sem:
        i = router.pick(key)
        router.inflight[i] += 1
        try:
            r = await clients[i].completions.create(model=model, prompt=prefix, max_tokens=max_tokens, temperature=1.0,
                                                    extra_body={"return_token_ids": True})
        except Exception as e:
            print(f"[resample] {key} failed: {e!r}", flush=True)
            return None
        finally:
            router.inflight[i] -= 1
    ch = r.choices[0]
    ids = getattr(ch, "token_ids", None) or (getattr(ch, "model_extra", None) or {}).get("token_ids")
    return ch.text or "", list(ids or []), getattr(ch, "finish_reason", None)


def target_of(gen: list, called: bool, rule: str, special: dict | None) -> tuple:
    """(loss tokens, None) of a sample, or (None, why); with 'cut' the tokens before the tool call."""
    if rule == "whole":
        return list(gen), None
    if gen.count(special["think_end"]) != 1:
        return None, "reasoning end not exactly once"
    end = gen.index(special["think_end"])
    if called:
        if special["call_open"] not in gen:
            return None, "first turn call token not found"
        k = gen.index(special["call_open"])
        if k < end:
            return None, "tool call before the reasoning ends"
        gen = gen[:k]
    elif special["call_open"] in gen:
        return None, "tool-call token without a parsed call"
    if special["call_close"] in gen:
        return None, "target holds a tool-call end"
    return list(gen), None


def choose(cid: str, lengths: list, ok: list, pick: str):
    """Index of the kept sample: the shortest passing one (ties to the lower index), or one drawn by a fixed hash."""
    good = [i for i, v in enumerate(ok) if v]
    if not good:
        return None
    if pick == "shortest":
        return min(good, key=lambda i: (lengths[i], i))
    return good[int(sha(f"random/{cid}"), 16) % len(good)]


async def run(a) -> dict:
    from openai import AsyncOpenAI

    from rci.agents.llm import Router

    out = Path(a.out)
    protocols = tuple(p for p in a.protocols.split(",") if p)
    special = special_ids(a.tokenizer) if a.target == "cut" else None
    submit = "keep" if a.target == "cut" else "reject"
    stats = Counter()
    dirs = protocol_dirs(a.protocol_dirs)
    with no_truth():
        bench, sets = load_bench(a.data.split(","))
        runs = a.harvest.split(",")
        jobs = contexts(runs, bench, dirs, protocols, seeds_of(a.seeds), stats)
        if not jobs:
            raise SystemExit(f"no captured first turns under {runs}: {dict(stats)}")
        f_prefix = {}
        if a.transplant_f:
            for c in sorted(contexts(runs, bench, dirs, ("f",), seeds_of(a.seeds), Counter()), key=lambda c: c["seed"]):
                f_prefix.setdefault((c["qid"], c["seat"]), c["prefix"])
        urls = a.base_urls.split(",")
        clients = [AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=3600, max_retries=2) for u in urls]
        router, sem = Router(len(urls)), asyncio.Semaphore(a.concurrency)
        kept = []

        async def one(c):
            got = await asyncio.gather(*(sample(clients, router, a.model, f"{c['cid']}#{k}", c["prefix"], a.max_tokens,
                                                sem) for k in range(a.k)))
            samples = []
            for g in got:
                verdict = judge(g, c["mine"], bench[c["qid"]], submit, a.criterion)
                called = bool(g) and has_tool_call(g[0])
                tgt, why = (None, verdict) if verdict not in ("ok",) else target_of(g[1], called, a.target, special)
                if tgt is not None and len(c["prefix"]) + len(tgt) > a.max_len:
                    tgt, why = None, "row longer than --max-len"
                samples.append({"verdict": verdict if why is None else why, "tool_call": called,
                                "finish": g[2] if g else None, "gen": g[1] if g else None, "target": tgt})
            kept.append({**c, "samples": samples})

        await asyncio.gather(*(one(c) for c in jobs))
    kept.sort(key=lambda k: k["cid"])
    rows, why = [], Counter()
    for k in kept:
        why.update(s["verdict"] for s in k["samples"])
        lengths = [len(s["target"] or []) for s in k["samples"]]
        i = choose(k["cid"], lengths, [s["target"] is not None for s in k["samples"]], a.pick)
        if i is None:
            continue
        tgt = k["samples"][i]["target"]
        rows.append({"qid": k["qid"], "protocol": k["proto"], "seed": k["seed"], "seat": k["seat"], "w": 1.0,
                     "ids": k["prefix"] + tgt, "spans": [[len(k["prefix"]), len(tgt)]]})
        if a.transplant_f and k["proto"] == "a":
            pre = f_prefix.get((k["qid"], k["seat"]))
            if pre is None:
                why["no free-debate prompt for the transplant"] += 1
            else:
                rows.append({"qid": k["qid"], "protocol": "f", "seed": k["seed"], "seat": k["seat"], "w": 1.0,
                             "ids": pre + tgt, "spans": [[len(pre), len(tgt)]]})
    digest = write_rows(out, rows)
    with gzip.open(out.with_suffix(".samples.jsonl.gz"), "wt") as f:
        for k in kept:
            f.write(json.dumps({"cid": k["cid"], "prefix_sha256": ids_hash(k["prefix"]),
                                "samples": [{x: s[x] for x in ("verdict", "tool_call", "finish", "gen")}
                                            for s in k["samples"]]}) + "\n")
    rep = {"label_free": True, "truth_guard": guard_report(), "sets": sets, "model": a.model, "k": a.k,
           "criterion": a.criterion, "target": a.target, "pick": a.pick, "transplant_f": a.transplant_f,
           "contexts": len(kept), **dict(stats), "samples": dict(why), "rows": len(rows),
           "loss_tokens": sum(n for r in rows for _, n in r["spans"]), "sha256": digest}
    out.with_suffix(".report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("contexts", "rows", "loss_tokens", "sha256")}))
    return rep


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sample K first turns per seat from a served model on the prompts of a "
                                             "captured harvest, judge each by the protocol's check, and keep the "
                                             "shortest passing one per seat as a training row.")
    ap.add_argument("--harvest", required=True, help="comma list of harvest roots of the parent generation")
    ap.add_argument("--data", required=True, help="comma list of task directories holding benchmark.json")
    ap.add_argument("--base-urls", required=True)
    ap.add_argument("--model", required=True, help="served name of the model that samples (the parent, or G0 for the "
                                                   "control)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--protocols", default="a,f")
    ap.add_argument("--protocol-dirs", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--criterion", choices=CRITERIA, default="loose")
    ap.add_argument("--target", choices=TARGETS, default="whole",
                    help="whole: a sample that calls a tool does not pass; cut: it may, and its target ends before the call")
    ap.add_argument("--tokenizer", default=None, help="model directory; needed with --target cut")
    ap.add_argument("--pick", choices=PICKS, default="shortest")
    ap.add_argument("--transplant-f", action="store_true",
                    help="also write each kept protocol-a turn after the free-debate prompt of the same seat")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-len", type=int, default=1 << 30, help="drop rows longer than this many tokens")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--out", required=True, help="output rows JSONL")
    a = ap.parse_args(argv)
    if a.target == "cut" and not a.tokenizer:
        ap.error("--target cut needs --tokenizer")
    asyncio.run(run(a))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
