"""Commit-turn rows: only the skeleton of a member's submission call is trained, weighted to a share of the loss."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rci.agents.capture import chains_path, read_chains
from rci.recursion.turns import (SUBMIT_TOOL, guard_report, harvest_records, load_bench, no_truth, protocol_dirs,
                                 seeds_of, sha)

LEVELS = ("on_time", "any_unforced")
WEIGHTINGS = ("loss", "rows")
ON_TIME_CALLS = 3
TOO_LONG = "row longer than --max-len"
OPEN, CLOSE, THINK_END = "<tool_call>", "</tool_call>", "</think>"
NAME = re.compile(r'<function=([^>\s<]+)|"name"\s*:\s*"([^"]+)"')
HEAD = ("", "{")
OVER = re.compile(r'[>"\s]*')
TAIL = re.compile(r"\s*(<\|[A-Za-z_]+\|>\s*)*")


def load_piece(path: str):
    """token id -> its text in the model's tokenizer (special tokens kept), memoised."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    memo: dict = {}

    def piece(t: int) -> str:
        if t not in memo:
            memo[t] = tok.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return memo[t]

    if piece(tok.convert_tokens_to_ids(OPEN)) != OPEN:
        raise SystemExit(f"{path}: the tokenizer has no {OPEN} token")
    return piece


def skeleton(gen: list, piece) -> tuple:
    """Token spans of a call skeleton (opening through function name, closing to end), or (None, why)."""
    pieces = [piece(t) for t in gen]
    ends = list(itertools.accumulate(len(p) for p in pieces))
    text = "".join(pieces)
    think = text.rfind(THINK_END)
    visible = think + len(THINK_END) if think >= 0 else 0
    opens = [i for i, p in enumerate(pieces) if p == OPEN and ends[i] - len(p) >= visible]
    if not opens:
        return None, "no call opening after the reasoning"
    o = opens[0]
    closes = [i for i in range(o + 1, len(pieces)) if pieces[i] == CLOSE]
    if not closes:
        return None, "no call closing after the opening"
    c = closes[-1]
    m = NAME.search(text, ends[o], ends[c] - len(CLOSE))
    if m is None:
        return None, "no function name between the opening and the closing"
    if text[ends[o]:m.start()].strip() not in HEAD:
        return None, "function name not at the head of the call"
    if not TAIL.fullmatch(text, ends[c]):
        return None, "text after the call's closing"
    g = 1 if m.group(1) is not None else 2
    last = bisect.bisect_left(ends, m.end(g))
    if not OVER.fullmatch(text, m.end(g), ends[last]):
        return None, "the name's last token runs into the parameters"
    return {"open": [o, last - o + 1], "close": [c, len(gen) - c], "name": m.group(g), "calls": len(opens)}, "ok"


def model_calls(agent: dict) -> list:
    return [c for c in agent.get("calls") or [] if isinstance(c, dict)]


def final_call(agent: dict, level: str) -> tuple:
    """Index of the call that gave the member's final answer when it is a candidate at `level`, else (None, why)."""
    calls = model_calls(agent)
    if agent.get("final_forced"):
        return None, "forced final"
    if level == "on_time" and len(calls) != ON_TIME_CALLS:
        return None, "late: more than three calls" if len(calls) > ON_TIME_CALLS else "early: fewer than three calls"
    if not calls or not (calls[-1].get("n_tool_calls") or 0) > 0:
        return None, "last call without a tool call"
    return len(calls) - 1, "ok"


def generations(chains: list) -> list:
    return [(i, j) for i, c in enumerate(chains) for j in range(len(c.get("spans") or []))]


def commit_row(qid: str, calls: list, target: int, chains: list, locate, max_len: int) -> tuple:
    """Where a candidate's row lies: its chain up to the final generation's end, and the two loss spans."""
    kept = [i for i, c in enumerate(calls) if not c.get("error") and c.get("tag") != "summary"]
    gens = generations(chains)
    if target not in kept or len(kept) != len(gens):
        return None, "captured generations do not match the calls"
    ci, si = gens[kept.index(target)]
    start, n = chains[ci]["spans"][si]
    gen = chains[ci]["ids"][start:start + n]
    sk, why = locate(gen)
    if sk is None:
        return None, why
    if sk["name"] != SUBMIT_TOOL:
        return None, "the call is not the submission"
    if start + n > max_len:
        return None, TOO_LONG
    (o, on), (c, cn) = sk["open"], sk["close"]
    return {"qid": qid, "at": [ci, start + n], "spans": [[start + o, on], [start + c, cn]],
            "open_id": gen[o], "close_id": gen[c]}, "ok"


def candidates(runs: list, proto_dir: str, seeds: set | None, bench: dict, level: str, locate, max_len: int,
               st: Counter) -> list:
    """Every member of the free-debate harvest on a task of the bench whose final call gives a commit row."""
    out = []
    for p, seed, rec in harvest_records(runs, proto_dir, 4, seeds):
        q = str(rec["example_id"])
        if q not in bench:
            continue
        st["records"] += 1
        chains = None
        for j, agent in enumerate(rec["agents"]):
            st["members"] += 1
            target, why = final_call(agent, level)
            if target is None:
                st[f"not a candidate: {why}"] += 1
                continue
            if not chains_path(Path(p)).exists():
                st["no row: record without chains"] += 1
                continue
            if chains is None:
                chains = read_chains(Path(p))
            cand, why = commit_row(q, model_calls(agent), target, chains.get(j, []), locate, max_len)
            if cand is None:
                st[f"no row: {why}"] += 1
            else:
                out.append({"key": f"{q}/a{j}/s{seed}", "path": p, "seat": j, **cand})
    return out


def commit_target(n_w: int, share: float) -> int:
    """How many commit rows make them `share` of all rows beside n_w other rows."""
    return int(n_w * share / (1.0 - share) + 0.5)


def draw(cands: list, n: int) -> list:
    """The first n candidates in a fixed hash order of (question, seat, seed)."""
    return sorted(cands, key=lambda c: sha(f"commit/{c['key']}"))[:n]


def commit_weight(weighting: str, share: float, w_loss: int, chosen: list) -> tuple:
    """(row weight of every commit row, their loss tokens): weight 1, or the weight giving them `share` of the loss."""
    c_loss = sum(n for c in chosen for _, n in c["spans"])
    if weighting == "rows" or not chosen:
        return 1.0, c_loss
    if c_loss <= 0 or w_loss <= 0:
        raise SystemExit("--weighting loss: the rows hold no loss token")
    return round(share / (1 - share) * w_loss / c_loss, 6), c_loss


def row_of(cand: dict, chains: list) -> dict:
    ci, end = cand["at"]
    ids = list(chains[ci]["ids"][:end]) if ci < len(chains) else []
    (o, _), (c, _) = cand["spans"]
    if len(ids) != end or ids[o] != cand["open_id"] or ids[c] != cand["close_id"]:
        raise SystemExit(f"{cand['key']}: the captured chain changed since the scan")
    return {"qid": cand["qid"], "protocol": "commit", "seat": cand["seat"], "w": 1.0, "ids": ids, "spans": cand["spans"]}


def read_rows(path: str) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(a) -> dict:
    piece = load_piece(a.tokenizer)
    w_rows = read_rows(a.w_rows) if a.w_rows else []
    w_loss = sum(n for r in w_rows for _, n in r["spans"])
    st = Counter()
    with no_truth():
        bench, sets = load_bench(a.data.split(","))
        cands = candidates(a.harvest.split(","), protocol_dirs(a.protocol_dirs)["f"], seeds_of(a.seeds), bench,
                           a.level, lambda gen: skeleton(gen, piece), a.max_len, st)
        n = commit_target(len(w_rows), a.share) if w_rows else len(cands)
        if len(cands) < n:
            raise SystemExit(f"{len(cands)} commit candidates, {n} needed: {dict(st)}")
        chosen = sorted(draw(cands, n), key=lambda c: c["key"])
        cw, c_loss = commit_weight(a.weighting, a.share, w_loss, chosen)
        cache: dict = {}

        def c_rows():
            for c in chosen:
                if c["path"] not in cache:
                    cache.clear()
                    cache[c["path"]] = read_chains(Path(c["path"]))
                yield {**row_of(c, cache[c["path"]].get(c["seat"], [])), "w": cw}

        digest = write(Path(a.out), itertools.chain(w_rows, c_rows()))
    rep = {"label_free": True, "truth_guard": guard_report(), "sets": sets, "level": a.level, "share": a.share,
           "weighting": a.weighting, "w_rows": len(w_rows), "w_loss_tokens": w_loss, "candidates": len(cands),
           "commit_rows": len(chosen), "commit_loss_tokens": c_loss, "row_weight": cw,
           "weighted_loss_share": round(cw * c_loss / (cw * c_loss + w_loss), 4) if chosen and w_loss else None,
           "reasons": dict(sorted(st.items())), "sha256": digest}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("w_rows", "commit_rows", "row_weight", "sha256")}))
    return rep


def concat(a) -> dict:
    rows = [r for p in a.inputs for r in read_rows(p)]
    digest = write(Path(a.out), rows)
    print(json.dumps({"rows": len(rows), "sha256": digest}))
    return {"rows": len(rows), "sha256": digest}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Commit-turn rows from a free-debate harvest, appended to first-turn rows; "
                                             "or the concatenation of row files.")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="first-turn rows followed by weighted commit-turn rows")
    b.add_argument("--harvest", required=True, help="comma list of harvest roots of the source model (free debate)")
    b.add_argument("--data", required=True, help="comma list of task directories holding benchmark.json")
    b.add_argument("--seeds", required=True, help="harvest seeds of this row set")
    b.add_argument("--protocol-dirs", default=None)
    b.add_argument("--w-rows", default=None, help="first-turn rows (rci.recursion.resample output) placed first")
    b.add_argument("--share", type=float, default=0.25, help="commit rows' share of all rows and of the loss")
    b.add_argument("--weighting", choices=WEIGHTINGS, default="loss")
    b.add_argument("--level", choices=LEVELS, default="on_time")
    b.add_argument("--max-len", type=int, default=8192)
    b.add_argument("--tokenizer", required=True, help="model directory whose tokenizer locates the call")
    b.add_argument("--out", required=True)
    c = sub.add_parser("concat", help="concatenate row files in the given order")
    c.add_argument("inputs", nargs="+")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.command == "build" and not 0 < a.share < 1:
        ap.error("--share must be between 0 and 1")
    (build if a.command == "build" else concat)(a)
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
