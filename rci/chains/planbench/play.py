"""Play the four-call PlanBench protocol against an OpenAI-compatible server; one record per (problem, seed).

    python -m rci.chains.planbench.play --problems FILE --set NAME --seeds 0 --model SERVED --base-urls URLS --out DIR
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
from pathlib import Path

from rci.chains.planbench import protocol as PR
from rci.chains.util import read_jsonl

SAMPLING = {"think": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "direct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0}}
MAX_TOKENS = {"draft": 8192, "check": 1024, "revise": 8192, "review": 3072}


def call_record(c) -> dict:
    return {"content": c.content or "", "fin": c.finish_reason, "ct": int(c.completion_tokens or 0),
            "pt": int(c.prompt_tokens or 0), "error": c.error, "prompt_ids": c.prompt_token_ids, "ids": c.token_ids,
            "think_text": bool(c.reasoning)}


async def ask(llm, text: str, call: int, seat: int, set_name: str, seed: int, key: str, max_tokens: int, mode: str):
    think = PR.thinks(call, mode)
    s = SAMPLING["think" if think else "direct"]
    c = await llm.chat([{"role": "user", "content": text}], max_tokens=max_tokens, temperature=s["temperature"],
                       top_p=s["top_p"], top_k=s["top_k"], presence_penalty=s.get("presence_penalty"),
                       seed=PR.request_seed(set_name, seed, key, seat, call), enable_thinking=think,
                       route_key=key, tag=f"call{call}", want_token_ids=True)
    return call_record(c)


async def play_game(llm, row: dict, set_name: str, seed: int, mode: str = "think", max_tokens: dict = None) -> dict:
    mt = dict(MAX_TOKENS, **(max_tokens or {}))
    p, key = row["query"], row["id"]
    seats = range(PR.SEATS)
    draft = await asyncio.gather(*[ask(llm, p, 1, s, set_name, seed, key, mt["draft"], mode) for s in seats])
    check = await asyncio.gather(*[ask(llm, PR.audit_prompt(p, [draft[s]["content"]]), 2, s, set_name, seed, key,
                                       mt["check"], mode) for s in seats])
    dt, ct = [x["content"] for x in draft], [x["content"] for x in check]
    final = await asyncio.gather(*[ask(llm, PR.revise_prompt(p, s, dt, ct), 3, s, set_name, seed, key, mt["revise"],
                                       mode) for s in seats])
    ft = [x["content"] for x in final]
    review = await asyncio.gather(*[
        ask(llm, PR.audit_prompt(p, [ft[t] for t in PR.jury_order(set_name, seed, key, a)]), 4, a, set_name, seed, key,
            mt["review"], mode) for a in seats])
    reviews = PR.reviews_of(set_name, seed, key, [x["content"] for x in review])
    return {"key": key, "set": set_name, "seed": seed, "mode": mode, "draft": draft, "check": check, "final": final,
            "review": review, "commit": PR.commit(reviews, set_name, seed, key)}


def record_path(out: Path, set_name: str, seed: int, key: str) -> Path:
    return out / set_name / f"s{seed}" / f"{key}.json.gz"


async def run(llm, rows: list, set_name: str, seeds: list, out: Path, concurrency: int, mode: str) -> dict:
    sem = asyncio.Semaphore(concurrency)
    done = {"played": 0, "kept": 0, "failed": 0}

    async def one(row, seed):
        p = record_path(out, set_name, seed, row["id"])
        if p.exists():
            done["kept"] += 1
            return
        async with sem:
            rec = await play_game(llm, row, set_name, seed, mode)
        if any(c["error"] for k in ("draft", "check", "final", "review") for c in rec[k]):
            done["failed"] += 1
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        with gzip.open(tmp, "wt") as f:
            json.dump(rec, f)
        os.replace(tmp, p)
        done["played"] += 1

    await asyncio.gather(*[one(r, s) for s in seeds for r in rows])
    return done


def main(argv=None) -> int:
    from rci.agents.llm import LLMClient

    ap = argparse.ArgumentParser(prog="python -m rci.chains.planbench.play", description=__doc__.splitlines()[0])
    ap.add_argument("--problems", required=True, help="JSONL of generated problems (rci.chains.planbench.generate)")
    ap.add_argument("--set", required=True, help="the set's name (enters every sampling seed)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-urls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="think", choices=("think", "direct"))
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args(argv)
    llm = LLMClient(a.base_urls.split(","), a.model, timeout=1800.0)
    rows = read_jsonl(a.problems)
    res = asyncio.run(run(llm, rows, a.set, [int(s) for s in a.seeds.split(",")], Path(a.out), a.concurrency, a.mode))
    print(json.dumps(res))
    return 3 if res["failed"] else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
