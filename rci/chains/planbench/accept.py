"""The PlanBench check: the plan simulator judges every draft and final, and accepted texts become training rows.

Rows of one problem q of a harvest game (r_q = the share of correct drafts):
    r_q = 1      archived: no row (a problem every seat solves teaches nothing new)
    no valid text among the four drafts and four finals: frontier, no row
    R1  up to k(r_q) distinct valid drafts as [user: P] -> draft (with its thought): the fewest actions first, then the
        completion length nearest the problem's median, then a hash order
    R2  at most one seat whose draft is wrong and whose final is valid: its revision message -> its final
    R3  at most two reviews, only when one final is right and one wrong, that are exact: every verdict parsed and equal
        to the simulator's, every INVALID naming the first failing step or the unmet goal
A valid text is strictly correct and clean, ends its plan before anything else, finished (not truncated), not empty
and carries no think tag. R2 and R3 are capped at 30 % and 15 % of a round's row tokens, and R1 rows are dropped from
the long end while their mean length exceeds 1.1 x their problems' medians (at most a quarter of them).
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import statistics
import zlib
from collections import Counter
from pathlib import Path

from rci.chains.planbench import domain as D
from rci.chains.planbench import protocol as PR
from rci.chains.planbench import text as T
from rci.chains.util import hkey

K_BY_RQ = {0.0: 2, 0.25: 2, 0.5: 2, 0.75: 1, 1.0: 0}


def load_records(root) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(str(root), "*", "s*", "*.json.gz"))):
        with gzip.open(p, "rt") as f:
            out.append(json.load(f))
    return out


def facts(x) -> frozenset:
    return frozenset(tuple(f) for f in x)


def degenerate(text: str) -> dict:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    rep = max(Counter(lines).values()) if lines else 0
    raw = (text or "").encode("utf-8")
    zr = len(zlib.compress(raw)) / len(raw) if raw else 1.0
    return {"rep": rep, "nch": len(text or ""), "zr": zr}


def judge_text(call: dict, row: dict) -> dict:
    v = T.judge(call["content"], facts(row["initial"]), facts(row["goal"]))
    v.update(ct=call["ct"], trunc=call["fin"] == "length", fin=call["fin"], opt=row["plan_length"],
             n=v["n_actions"], **degenerate(call["content"]))
    return v


def valid(v: dict, require_plan_end: bool = False) -> bool:
    if not (v["strict_ok"] and v["clean"]) or v["tail"] or v["trunc"] or v["empty"] or v["think_tag"]:
        return False
    return v["end"] or not require_plan_end


def review_exact(text: str, finals: list, row: dict, rec: dict, reviewer: int) -> bool:
    got = PR.reviews_of(rec["set"], rec["seed"], rec["key"], [""] * reviewer + [text])[reviewer]
    for t, v in got.items():
        if v is None:
            return False
        kind, step = D.first_failure(facts(row["initial"]), facts(row["goal"]), [tuple(a) for a in finals[t]["plan"]])
        if v["valid"] != (kind == "valid"):
            return False
        if not v["valid"] and (v["why"] != kind or (kind == "step" and v["step"] != step)):
            return False
    return True


def judge_record(rec: dict, row: dict) -> dict:
    """Every verdict of one game; `commit` is the committed final's strict correctness."""
    draft = [judge_text(c, row) for c in rec["draft"]]
    final = [judge_text(c, row) for c in rec["final"]]
    reviews = PR.reviews_of(rec["set"], rec["seed"], rec["key"], [c["content"] for c in rec["review"]])
    checks = [PR.parse_verdicts(c["content"], "A")["A"] for c in rec["check"]]
    leaks = [v["think_tag"] for v in draft + final] + [c["think_text"] or T.has_think_tag(c["content"])
                                                       for c in rec["check"] + rec["review"]]
    t = rec["commit"]["t"]
    return {"key": rec["key"], "seed": rec["seed"], "draft": draft, "final": final,
            "commit": {"ok": final[t]["strict_ok"], "ct": rec["final"][t]["ct"], "t": t},
            "oracle": any(v["strict_ok"] for v in final),
            "parse_verdict": [x is not None for x in checks] + [x is not None for rv in reviews.values()
                                                                for x in rv.values()],
            "leak": leaks, "review_trunc": [c["fin"] == "length" for c in rec["review"]]}


def judged(root, problems: dict) -> list:
    out = []
    for rec in load_records(root):
        if rec["key"] in problems:
            out.append((rec, judge_record(rec, problems[rec["key"]])))
    return out


def mean(xs, default=float("nan")) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else default


def stats(lines: list, h: dict) -> dict:
    """The health quantities and the rates of one cell (list of judge_record outputs)."""
    texts = [x for v in lines for x in v["draft"] + v["final"]]
    ok = [x for x in texts if x["strict_ok"]]
    return {
        "n": len(lines),
        "rate": mean(v["commit"]["ok"] for v in lines),
        "seat_draft": mean(x["strict_ok"] for v in lines for x in v["draft"]),
        "seat_final": mean(x["strict_ok"] for v in lines for x in v["final"]),
        "oracle": mean(v["oracle"] for v in lines),
        "draft_ct": mean(x["ct"] for v in lines for x in v["draft"]),
        "commit_ct": mean(v["commit"]["ct"] for v in lines),
        "empty_fin": mean(bool(x["empty"]) and not x["trunc"] for x in texts),
        "trunc": mean(x["trunc"] for x in texts),
        "parse": mean([x["n"] > 0 for x in texts] + [p for v in lines for p in v["parse_verdict"]], 1.0),
        "degenerate": mean(x["rep"] >= h["H7_line_repeats"] or (x["nch"] >= h["H7_min_chars"] and x["zr"] < h["H7_zr"])
                           for x in texts),
        "think": mean((p for v in lines for p in v["leak"]), 0.0),
        "runaway": (statistics.median(x["n"] for x in ok) / statistics.median(x["opt"] for x in ok)) if ok else None,
        "steps_over_optimal": (sum(x["n"] for x in ok) / sum(x["opt"] for x in ok)) if ok else None,
    }


def health(cand: dict, base: dict, h: dict) -> list:
    from rci.chains import health as HL
    return [
        HL.within("H1 draft tokens / base", HL.ratio(cand["draft_ct"], base["draft_ct"]), *h["H1"]),
        HL.within("H2 committed tokens / base", HL.ratio(cand["commit_ct"], base["commit_ct"]), *h["H2"]),
        HL.at_most("H3 empty finished answers", cand["empty_fin"], max(h["H3_abs"], base["empty_fin"] + h["H3_add"])),
        HL.at_most("H4 truncated answers", cand["trunc"], base["trunc"] + h["H4_add"]),
        HL.at_least("H6 parse rate", cand["parse"], base["parse"] - h["H6_sub"]),
        HL.at_most("H7 degenerate repetition", cand["degenerate"], base["degenerate"] + h["H7_add"]),
        HL.at_most("H8 thinking leak", cand["think"], max(h["H8_abs"], base["think"] + h["H8_add"])),
        HL.Line("HR runaway plans", cand["runaway"], h["runaway_ratio"],
                cand["runaway"] is None or cand["runaway"] <= h["runaway_ratio"] + HL.EPS),
    ]


def plan_problem(rec: dict, v: dict, r: int, c: dict) -> dict:
    drafts, finals = v["draft"], v["final"]
    rq = sum(x["strict_ok"] for x in drafts) / PR.SEATS
    texts = [("draft", s, drafts[s], rec["draft"][s]) for s in range(PR.SEATS)] + \
            [("final", s, finals[s], rec["final"][s]) for s in range(PR.SEATS)]
    if rq == 1.0:
        return {"cls": "archived", "rows": []}
    if not any(valid(x[2], c["require_plan_end"]) for x in texts):
        return {"cls": "frontier", "rows": []}
    cand = [t for t in texts if t[0] in c["r1_sources"]]
    good = [t for t in cand if valid(t[2], c["require_plan_end"])]
    med = statistics.median([call["ct"] for *_, call in cand]) or 1
    seen, r1 = set(), []
    for src, s, vv, call in sorted(good, key=lambda t: (t[2]["n"], abs(t[3]["ct"] - med), 0 if t[0] == "final" else 1,
                                                        hkey("planbench", "row", r, rec["key"], t[1], t[0]))):
        if len(r1) >= K_BY_RQ[round(rq, 2)]:
            break
        if call["content"] in seen:
            continue
        seen.add(call["content"])
        r1.append({"kind": "R1", "src": src, "seat": s, "len": call["ct"], "ratio": call["ct"] / med,
                   "h": hkey("planbench", "row", r, rec["key"], s, src)})
    r2 = []
    for s in sorted(range(PR.SEATS), key=lambda s: hkey("planbench", "r2", r, rec["key"], s)):
        if len(r2) >= c["r2_per_q"]:
            break
        if not drafts[s]["strict_ok"] and valid(finals[s], c["require_plan_end"]):
            r2.append({"kind": "R2", "src": "final", "seat": s, "len": rec["final"][s]["ct"],
                       "h": hkey("planbench", "r2", r, rec["key"], s)})
    r3 = []
    fin_ok = [x["strict_ok"] for x in finals]
    if any(fin_ok) and not all(fin_ok):
        for s in sorted(range(PR.SEATS), key=lambda s: hkey("planbench", "r3", r, rec["key"], s)):
            if len(r3) >= c["r3_per_q"]:
                break
            call = rec["review"][s]
            if call["fin"] != "stop" or call["think_text"] or T.has_think_tag(call["content"]):
                continue
            if review_exact(call["content"], finals, rec["_row"], rec, s):
                r3.append({"kind": "R3", "src": "review", "seat": s, "len": call["ct"],
                           "h": hkey("planbench", "r3", r, rec["key"], s)})
    return {"cls": "trainable", "rows": r1 + r2 + r3}


def length_rule(r1: list, hi: float, max_share: float) -> list:
    keep = sorted(r1, key=lambda x: (x["ratio"], x["h"]))
    budget, gone = int(max_share * len(keep)), []
    while len(keep) > 1 and len(gone) < budget and sum(x["ratio"] for x in keep) / len(keep) > hi:
        gone.append(keep.pop())
    return gone


def plan_round(pairs: list, r: int, c: dict) -> tuple:
    planned, counts = [], Counter()
    for rec, v in pairs:
        pp = plan_problem(rec, v, r, c)
        counts[pp["cls"]] += 1
        for x in pp["rows"]:
            x["rec"] = rec
        planned += pp["rows"]
    gone = {id(x) for x in length_rule([x for x in planned if x["kind"] == "R1"], c["len_hi"], c["len_drop_max_share"])}
    planned = [x for x in planned if id(x) not in gone]
    capped = Counter()
    for kind in ("R3", "R2"):
        cap = float(c["share_cap"][kind])
        rows_k = sorted((x for x in planned if x["kind"] == kind), key=lambda x: x["h"])
        while rows_k and sum(x["len"] for x in rows_k) > cap * sum(x["len"] for x in planned):
            out = rows_k.pop()
            planned = [x for x in planned if x is not out]
            capped[kind] += 1
    return planned, {"classes": dict(counts), "rows": dict(Counter(x["kind"] for x in planned)),
                     "length_dropped": len(gone), "share_capped": dict(capped)}


def encode(planned: list, eos_id: int, max_tokens: int) -> list:
    out = []
    for x in planned:
        rec = x["rec"]
        call = {"R1": rec["draft"], "R2": rec["final"], "R3": rec["review"]}[x["kind"]][x["seat"]]
        if not call.get("prompt_ids") or not call.get("ids"):
            continue
        comp = list(call["ids"])
        if comp[-1] != eos_id:
            comp.append(eos_id)
        ids = list(call["prompt_ids"]) + comp
        if len(ids) > max_tokens:
            continue
        out.append({"row_id": f"r{rec['_round']}:{rec['key']}:s{rec['seed']}:{x['kind']}:{x['seat']}", "ids": ids,
                    "spans": [[len(call["prompt_ids"]), len(comp)]], "w": 1.0, "kind": x["kind"]})
    return out


def eos_of(tokenizer_dir: str) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    t = tok.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(t, int) or t < 0 or t == tok.unk_token_id:
        raise SystemExit(f"{tokenizer_dir}: no <|im_end|> token")
    return t


def build_rows(records_root, problems: dict, r: int, c: dict, eos_id: int) -> tuple:
    pairs = judged(records_root, problems)
    for rec, _ in pairs:
        rec["_row"], rec["_round"] = problems[rec["key"]], r
    planned, meta = plan_round(pairs, r, c)
    rows = encode(planned, eos_id, c["row_max_tokens"])
    meta.update(problems=len(pairs), encoded=len(rows), tokens=sum(len(x["ids"]) for x in rows))
    return rows, meta


def read_problems(path) -> dict:
    from rci.chains.util import read_jsonl
    return {x["id"]: x for x in read_jsonl(path)}


def problems_file(sets_dir, name: str) -> Path:
    return Path(sets_dir) / f"{name}.jsonl"
