"""First-generation rows: the verified first turns of a captured harvest, balanced over questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rci.agents.capture import chains_path, read_chains
from rci.recursion.turns import (SUBMIT_TOOL, call_reason, calls_of, guard_report, harvest_records, ids_hash,
                                 load_bench, mine_of, no_truth, protocol_dirs, seeds_of, sha, special_ids, target_len,
                                 text_of)
from rci.recursion.verify import first_span_row, first_turn_ok

SUBMIT_TURNS = ("reject", "keep", "cut")


def scan(runs: list, bench: dict, dirs: dict, protocols: tuple, seeds: set | None, submit: str, call_id: int | None,
         trained: set, stats: dict) -> list:
    """Every member's first turn on a task of the bench, judged by the protocol's check (no label is read)."""
    out = []
    for proto in protocols:
        st = stats.setdefault(proto, Counter())
        for p, seed, rec in harvest_records(runs, dirs[proto], 4, seeds):
            q = str(rec["example_id"])
            row = bench.get(q)
            if row is None:
                st["record on no task of the bench"] += 1
                continue
            st["records"] += 1
            if not chains_path(Path(p)).exists() or not rec.get("transcripts"):
                st["record without chains or transcripts"] += 1
                continue
            chains = read_chains(Path(p))
            n_ag = len(rec["agents"])
            for aid, turns in sorted(rec["transcripts"].items(), key=lambda x: int(x[0])):
                j = int(aid)
                mine = mine_of(row, n_ag, j)
                if not mine or not turns:
                    st["seat without private facts"] += 1
                    continue
                said = [m for m in turns[0].get("messages", []) if m.get("role") == "assistant"]
                calls = calls_of(said)
                if len(said) == 1:
                    ok, reason = first_turn_ok(text_of(said[0]), mine, row)
                    if ok and calls and submit == "reject":
                        ok, reason = False, call_reason(calls)
                else:
                    ok, reason = False, ("first turn with several generations" if said else "first turn without a generation")
                r = first_span_row(chains.get(j, []))
                if r is None:
                    st["no captured first span"] += 1
                    continue
                if ids_hash(r["ids"]) in trained:
                    st["row already trained (excluded)"] += 1
                    continue
                start, n = r["spans"][0]
                cut = target_len(r["ids"], start, n, bool(calls), submit, call_id)
                if cut is None:
                    if ok:
                        ok, reason = False, "first turn call token not found"
                    cut = n
                st[f"reason: {reason}"] += 1
                out.append({"key": f"{proto}/{q}/s{seed}/a{j}", "proto": proto, "qid": q, "seed": seed, "seat": j,
                            "path": p, "ok": ok, "n": cut, "called": bool(calls)})
    return out


def allot(caps: dict, total: int) -> dict:
    """One more row per question per pass, questions in a fixed hash order, until `total` or every cap is reached."""
    order = sorted(caps, key=lambda q: sha(f"allot/{q}"))
    got = {q: 0 for q in caps}
    left = total
    for level in range(1, max(caps.values(), default=0) + 1):
        for q in order:
            if left <= 0:
                return got
            if caps[q] >= level:
                got[q] += 1
                left -= 1
    return got


def select(turns: list, protocols: tuple, limits: dict) -> list:
    """The qualifying turns kept per protocol: all of them, or a water-filled allotment of `limits[proto]` rows."""
    out = []
    for proto in protocols:
        by_q = defaultdict(list)
        for t in turns:
            if t["proto"] == proto and t["ok"]:
                by_q[t["qid"]].append(t)
        caps = {q: len(v) for q, v in by_q.items()}
        k = allot(caps, limits[proto]) if limits.get(proto) else caps
        for q in sorted(by_q):
            out += sorted(by_q[q], key=lambda t: sha(f"rows/{t['key']}"))[:k[q]]
    return out


def row_of(t: dict, cache: dict) -> dict:
    if t["path"] not in cache:
        cache.clear()
        cache[t["path"]] = read_chains(Path(t["path"]))
    r = first_span_row(cache[t["path"]].get(t["seat"], []))
    start, n = r["spans"][0]
    if t["n"] != n:
        r = {"ids": r["ids"][:start + t["n"]], "spans": [[start, t["n"]]]}
    return {"qid": t["qid"], "protocol": t["proto"], "seed": t["seed"], "seat": t["seat"], "w": 1.0, **r}


def write_rows(path: Path, rows: list) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(data)
    return hashlib.sha256(data.encode()).hexdigest()


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build first-generation training rows from a captured harvest: first "
                                             "turns that write every private fact of the member and state no verdict.")
    ap.add_argument("--harvest", required=True, help="comma list of harvest roots (rci.run --out of the capture runs)")
    ap.add_argument("--data", required=True, help="comma list of task directories holding benchmark.json")
    ap.add_argument("--protocols", default="a,f", help="a: debate with fact forwarding; f: three-round free debate")
    ap.add_argument("--protocol-dirs", default=None, help="override the record directories, e.g. a=DIR,f=DIR")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--submit-turns", choices=SUBMIT_TURNS, default="reject",
                    help="a passing first turn that also calls a tool: reject it, keep it, or cut its target before the call")
    ap.add_argument("--tokenizer", default=None, help="model directory; needed with --submit-turns cut")
    ap.add_argument("--limit-a", type=int, default=1700, help="rows of protocol a, water-filled over questions (0: all)")
    ap.add_argument("--limit-f", type=int, default=1700, help="rows of protocol f (0: all)")
    ap.add_argument("--trained", action="append", default=[], help="row files already trained on; their rows are skipped")
    ap.add_argument("--out", required=True, help="output rows JSONL; a report is written beside it")
    a = ap.parse_args(argv)
    protocols = tuple(p for p in a.protocols.split(",") if p)
    if a.submit_turns == "cut" and not a.tokenizer:
        ap.error("--submit-turns cut needs --tokenizer")
    call_id = special_ids(a.tokenizer)["call_open"] if a.submit_turns == "cut" else None
    with no_truth():
        bench, sets = load_bench(a.data.split(","))
        trained = set()
        for f in a.trained:
            trained |= {ids_hash(json.loads(x)["ids"]) for x in Path(f).read_text().splitlines() if x.strip()}
        stats: dict = {}
        turns = scan(a.harvest.split(","), bench, protocol_dirs(a.protocol_dirs), protocols, seeds_of(a.seeds),
                     a.submit_turns, call_id, trained, stats)
        chosen = select(turns, protocols, {"a": a.limit_a, "f": a.limit_f})
        if not chosen:
            raise SystemExit(f"no qualifying first turn: {json.dumps({k: dict(v) for k, v in stats.items()})}")
        cache: dict = {}
        rows = [row_of(t, cache) for t in sorted(chosen, key=lambda t: (t["proto"], t["qid"], t["key"]))]
    out = Path(a.out)
    digest = write_rows(out, rows)
    rep = {"label_free": True, "truth_guard": guard_report(), "sets": sets, "submit_turns": a.submit_turns,
           "submit_tool": SUBMIT_TOOL,
           "protocols": {p: {"first_turns": sum(t["proto"] == p for t in turns),
                             "qualifying": sum(t["proto"] == p and t["ok"] for t in turns),
                             "rows": sum(r["protocol"] == p for r in rows),
                             "reasons": dict(sorted(stats.get(p, Counter()).items()))} for p in protocols},
           "rows": len(rows), "loss_tokens": sum(n for r in rows for _, n in r["spans"]), "sha256": digest}
    out.with_suffix(".report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("rows", "loss_tokens", "sha256")}))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
