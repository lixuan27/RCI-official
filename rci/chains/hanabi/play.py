"""Play Hanabi against an OpenAI-compatible vLLM server: one /v1/completions request per move, the prompt sent as token
ids rendered with the model's own chat template (thinking off), the output constrained to the legal move codes.

    python -m rci.chains.hanabi.play --role train --round 3 --deals LO HI --games 4 --model SERVED
                                     --base-urls URLS --tokenizer DIR --out DIR

A record per game, <out>/<deal>_<g>.json. An output outside the legal codes, a finish other than "stop" or more than
max_completion_tokens tokens ends the game with score 0 and end "protocol" (a record); a server failure after the
retries writes no record, and the run exits 3.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

from rci.chains.hanabi import engine as E
from rci.chains.hanabi import obs as O

STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")
REQUEST = {"max_tokens": 8, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "max_completion_tokens": 5, "retries": 4,
           "backoff_s": (2, 4, 8, 16), "timeout_s": 120}
ROLES = ("train", "select", "confirm")
BASES = {"train": 2_000_000, "select": 3_000_000, "confirm": 4_000_000}
STRIDE = 10_000


class InfraError(RuntimeError):
    pass


def deal_range(role: str, r: int, n: int) -> list:
    """[lo, hi] of round r's fresh deals of a role: disjoint across roles and rounds."""
    if not 1 <= n <= STRIDE:
        raise ValueError(f"{n} deals do not fit one round's stride")
    lo = BASES[role] + STRIDE * r
    return [lo, lo + n - 1]


def samp_seed(role: str, r: int, deal: int, g: int, turn: int) -> int:
    """Depends on (role, round, deal, game, turn) only: every model meets the same random numbers on a deal."""
    h = hashlib.sha256(f"hanabi/samp/{role}/{r}/{deal}/{g}/{turn}".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


class Http:
    def __init__(self, timeout: float, retries: int, backoff):
        import httpx
        self.client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=4096), trust_env=False)
        self.retries, self.backoff = retries, list(backoff)

    async def post(self, url: str, body: dict) -> dict:
        last = None
        for a in range(self.retries + 1):
            try:
                r = await self.client.post(url, json=body)
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                    raise InfraError(last)
            except InfraError:
                raise
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
            if a < self.retries:
                await asyncio.sleep(self.backoff[min(a, len(self.backoff) - 1)])
        raise InfraError(f"{url}: {last}")


def check(resp: dict, ids: list, legal: list, max_ct: int):
    """((code, completion ids, logprob sum or None), None) or (None, why)."""
    try:
        ch = resp["choices"][0]
    except (KeyError, IndexError, TypeError):
        return None, "no choice in the response"
    text = ch.get("text") or ""
    for s in STOP_STRINGS:
        text = text.replace(s, "")
    text = text.strip()
    toks = ch.get("token_ids") or []
    pids = ch.get("prompt_token_ids") or resp.get("prompt_token_ids")
    if pids is not None and [int(x) for x in pids] != ids:
        return None, "the echoed prompt ids differ from the ids sent"
    if text not in legal:
        return None, f"output {text[:40]!r} is not a legal code"
    if ch.get("finish_reason") != "stop":
        return None, f"finish_reason {ch.get('finish_reason')!r}"
    if not toks or len(toks) > max_ct:
        return None, f"{len(toks)} completion tokens"
    lps = (ch.get("logprobs") or {}).get("token_logprobs") if isinstance(ch.get("logprobs"), dict) else None
    lp = float(sum(x for x in lps if x is not None)) if lps else None
    return (text, [int(x) for x in toks], lp), None


async def play_game(post, tok, url: str, model: str, role: str, r: int, deal: int, g: int, temperature: float,
                    req: dict = REQUEST) -> dict:
    s = E.new_game(deal)
    moves, proto = [], None
    while s.end is None:
        ids = O.prompt_ids(tok, O.observe(s))
        legal = E.legal_codes(s)
        body = {"model": model, "prompt": ids, "max_tokens": req["max_tokens"], "n": 1, "temperature": temperature,
                "top_p": req["top_p"], "top_k": req["top_k"], "min_p": req["min_p"], "logprobs": 1,
                "return_token_ids": True, "seed": samp_seed(role, r, deal, g, s.turn),
                "structured_outputs": {"choice": legal}}
        got, why = check(await post(url.rstrip("/") + "/completions", body), ids, legal, req["max_completion_tokens"])
        if got is None:
            proto = why
            break
        code, toks, lp = got
        mv = {"t": s.turn, "s": s.cur, "c": code, "tok": toks, "sha": O.ids_sha(ids), "pl": len(ids)}
        if lp is not None:
            mv["lp"] = round(lp, 5)
        moves.append(mv)
        s = E.apply_code(s, code)
    return {"role": role, "round": r, "deal": deal, "g": g, "T": temperature, "model": model, "moves": moves,
            "end": "protocol" if proto else s.end, "protocol": proto, "s_ch": 0 if proto else E.score_ch(s),
            "s_pl": 0 if proto else E.score_pl(s), "lives_left": s.lives, "n_moves": len(moves),
            "n_hints": sum(m["c"][0] == "H" for m in moves), "system_sha256": O.SYSTEM_SHA256}


async def run(post, tok, urls: list, model: str, role: str, r: int, deals: list, games: int, temperature: float,
              out: Path, concurrency: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    todo = [(d, g) for d in range(deals[0], deals[1] + 1) for g in range(games)
            if not (out / f"{d}_{g}.json").exists()]
    q: asyncio.Queue = asyncio.Queue()
    for x in todo:
        q.put_nowait(x)
    stats = {"to_play": len(todo), "played": 0, "protocol": 0, "infra": 0}

    async def worker():
        while True:
            try:
                d, g = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                rec = await play_game(post, tok, urls[(7 * d + g) % len(urls)], model, role, r, d, g, temperature)
            except InfraError:
                stats["infra"] += 1
                continue
            p = out / f"{d}_{g}.json"
            tmp = p.with_name(p.name + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(rec, separators=(",", ":")))
            os.replace(tmp, p)
            stats["played"] += 1
            stats["protocol"] += rec["end"] == "protocol"

    await asyncio.gather(*[worker() for _ in range(max(1, min(len(todo), concurrency * len(urls))))])
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rci.chains.hanabi.play", description=__doc__.splitlines()[0])
    ap.add_argument("--role", required=True, choices=ROLES)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--deals", type=int, nargs=2, required=True, metavar=("LO", "HI"))
    ap.add_argument("--games", type=int, default=1, help="games a deal")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-urls", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=128, help="games in flight a server")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    http = Http(REQUEST["timeout_s"], REQUEST["retries"], REQUEST["backoff_s"])
    st = asyncio.run(run(http.post, tok, a.base_urls.split(","), a.model, a.role, a.round, a.deals, a.games,
                         a.temperature, Path(a.out), a.concurrency))
    print(json.dumps(st))
    return 3 if st["infra"] else 0


if __name__ == "__main__":
    sys.exit(main())
