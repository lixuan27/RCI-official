"""Token accounting of blackboard records: waiting, negotiation and work, and the budget without waiting."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rci.ladder.commit import _candidates, vote_stake
from rci.ladder.curves import SIZES, at, members_of, pass_curves
from rci.ladder.records import records

FORWARD_TITLE = "I've forwarded facts I hold, word for word"
CLASSES = ("idle", "neg", "work", "other", "summary")
Z = 1.96


def protocol_kwargs(rec: dict) -> dict:
    cfg = rec.get("config") or {}
    return cfg.get("protocol_kwargs") or cfg.get("method_kwargs") or {}


def ms(xs: list) -> tuple:
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1) / len(xs))


def ratio_se(num_q: dict, den_q: dict) -> tuple:
    """A share summed over question clusters, with its linearised clustered standard error."""
    qs = [q for q in den_q if den_q[q] > 0]
    num, den = sum(num_q[q] for q in qs), sum(den_q[q] for q in qs)
    if den <= 0:
        return float("nan"), float("nan")
    r = num / den
    g = len(qs)
    var = sum((num_q[q] - r * den_q[q]) ** 2 for q in qs) * g / max(g - 1, 1) / den ** 2
    return r, math.sqrt(var)


def opens_of(kw: dict, seat: int) -> int:
    """The first step at which a seat may post or read outside the planning round."""
    by_seat_ = kw.get("planning_steps_by_seat")
    if by_seat_:
        rb = kw.get("read_budget") or []
        b = rb[min(seat, len(rb) - 1)] if rb else 0
        if b < 0:
            return 0
        return int(by_seat_[min(seat, len(by_seat_) - 1)])
    return int(kw.get("planning_steps") or 0)


def _turn_class(kinds: list) -> str:
    if not kinds:
        return "idle"
    if "work" in kinds:
        return "work"
    if "neg" in kinds:
        return "neg"
    if all(k == "idle" for k in kinds):
        return "idle"
    return "other"


def ledger(rec: dict) -> dict:
    """Completion tokens of one record by turn class, polling tokens, and tokens of every waiting turn."""
    kw = protocol_kwargs(rec)
    state = rec.get("method_state") or {}
    holds = defaultdict(list)
    for h in state.get("holds", []) or []:
        holds[(h["agent"], h["step"])].append(h["kind"])
    seen_ev = defaultdict(list)
    for e in state.get("events", []) or []:
        op = e.get("op")
        if op == "push" and e.get("view") == "full":
            seen_ev[e["reader"]].append((e["step"], 0, "push", set(e.get("ids") or [])))
        elif op == "search":
            seen_ev[e.get("reader")].append((e["step"], 1, "search", {r["id"] for r in e.get("results") or []}))
        elif op == "visit":
            seen_ev[e.get("reader")].append((e["step"], 1, "visit", set(e.get("ids") or [])))
    for m in state.get("messages", []) or []:
        seen_ev[m["author"]].append((m["step"], 2, "own", {m["id"]}))
    out = {"comp": Counter(), "turns": Counter(), "poll": 0, "mismatch": 0, "drop_wait_completion": 0,
           "drop_wait_total": 0}
    for a in rec.get("agents") or []:
        aid = a["id"]
        calls = a.get("calls") or []
        turn_calls = [i for i, c in enumerate(calls) if c.get("tag") in ("turn", "deliberate")]
        real = [t for t in a.get("turns") or [] if t.get("tools") != ["forced_final"]]
        if len(turn_calls) != len(real):
            out["mismatch"] += 1
        opens = opens_of(kw, aid)
        evs = sorted(seen_ev.get(aid, []), key=lambda x: (x[0], x[1]))
        search_ids = defaultdict(set)
        for st, _, kind, ids in evs:
            if kind == "search":
                search_ids[st] |= ids

        def shown_before(step):
            got = set()
            for st, _, kind, ids in evs:
                if st < step or (st == step and kind == "push"):
                    got |= ids
            return got

        cls_of_call = {}
        for t, ci in zip(real, turn_calls):
            s, tools = t["step"], t.get("tools") or []
            held = Counter(holds.get((aid, s), []))["submit"]
            kinds = []
            for name in tools:
                if name == "wait":
                    kinds.append("idle")
                elif name == "submit_answer":
                    if held > 0:
                        kinds.append("idle")
                        held -= 1
                    else:
                        kinds.append("work")
                elif name == "give_reading_budget":
                    kinds.append("neg")
                elif name in ("forward_facts", "publish_message", "visit_messages"):
                    kinds.append("neg" if s < opens else "work")
                elif name == "search_datastore":
                    kinds.append("work")
                else:
                    kinds.append("other")
            cls = _turn_class(kinds)
            cls_of_call[ci] = cls
            out["turns"][cls] += 1
            work_names = [n for n, k in zip(tools, kinds) if k == "work"]
            if (cls == "work" and work_names and all(n == "search_datastore" for n in work_names)
                    and not (search_ids.get(s, set()) - shown_before(s))):
                out["poll"] += int(calls[ci].get("completion_tokens") or 0)
        for i, c in enumerate(calls):
            k = cls_of_call.get(i, "summary" if c.get("tag") == "summary" else "work")
            out["comp"][k] += int(c.get("completion_tokens") or 0)
        dropped = {ci for ci, k in cls_of_call.items() if k == "idle"}
        carried = 0
        for i, c in enumerate(calls):
            p, cc = int(c.get("prompt_tokens") or 0), int(c.get("completion_tokens") or 0)
            if i in dropped:
                out["drop_wait_completion"] += cc
                out["drop_wait_total"] += p + cc
                carried += cc
            else:
                out["drop_wait_total"] += min(carried, p)
                if c.get("tag") == "summary":
                    carried = 0
    return out


def shares(rec: dict) -> dict | None:
    """Waiting, negotiation, polling and coordination shares of one record's completion tokens."""
    if not rec.get("agents") or not any(a.get("calls") for a in rec["agents"]):
        return None
    led = ledger(rec)
    if led["mismatch"] >= len(rec["agents"]):
        return None
    total = sum(led["comp"].values())
    if total <= 0:
        return None
    idle, neg, poll = led["comp"]["idle"], led["comp"]["neg"], led["poll"]
    return {"wait": idle / total, "neg": neg / total, "poll": poll / total, "coord": (idle + neg) / total,
            "coord_with_poll": (idle + neg + poll) / total, "completion": total}


def account(recs: dict, c1: dict, t_curves: dict, common: list) -> dict:
    """Shares, budgets and margins of one cell, now and with every waiting turn removed."""
    led = {q: [ledger(r) for r in rr] for q, rr in recs.items() if q in t_curves}
    comp_q = {q: sum(sum(x["comp"].values()) for x in v) for q, v in led.items()}
    res = {"records": sum(len(v) for v in led.values()), "questions": len(led)}
    for k in CLASSES:
        res[f"share_{k}"] = ratio_se({q: sum(x["comp"][k] for x in v) for q, v in led.items()}, comp_q)
    res["share_poll"] = ratio_se({q: sum(x["poll"] for x in v) for q, v in led.items()}, comp_q)
    res["share_coord"] = ratio_se({q: sum(x["comp"]["idle"] + x["comp"]["neg"] for x in v) for q, v in led.items()},
                                  comp_q)
    ys = {q: sum(vote_stake(r) for r in recs[q]) / len(recs[q]) for q in led}
    res["answer"] = ms([ys[q] for q in sorted(ys)])
    costs = {"completion_now": [], "completion_free_wait": [], "total_now": [], "total_free_wait": []}
    qidx = []
    for q, v in led.items():
        for r, x in zip(recs[q], v):
            comp = float((r.get("compute") or {}).get("completion_tokens") or 0)
            tot = float((r.get("compute") or {}).get("total_tokens") or 0)
            costs["completion_now"].append(comp)
            costs["completion_free_wait"].append(comp - x["drop_wait_completion"])
            costs["total_now"].append(tot)
            costs["total_free_wait"].append(tot - x["drop_wait_total"])
            qidx.append(q)
    qs = sorted(ys)
    for k, vals in costs.items():
        unit = k.split("_")[0]
        nb = sum(vals) / len(vals) / c1[unit]
        per_q = defaultdict(list)
        for q, x in zip(qidx, vals):
            per_q[q].append(x / c1[unit])
        e = ms([ys[q] - at({s: t_curves[q][s] for s in common}, nb) for q in qs])
        cls = ("capped" if nb > common[-1] else "above" if e[0] - Z * e[1] > 0
               else "below" if e[0] + Z * e[1] < 0 else "undetermined")
        res[f"n_budget_{k}"] = (nb, ms([sum(z) / len(z) for z in per_q.values()])[1])
        res[f"margin_{k}"] = (e[0], e[1], cls)
    return res


def paired_shares(x: dict, r: dict, key: str) -> dict:
    """X - R of one share, paired by (question, seed) and averaged per question."""
    per_q: dict = {}
    for (q, s), v in x.items():
        if (q, s) in r:
            per_q.setdefault(q, []).append(v[key] - r[(q, s)][key])
    if len(per_q) < 2:
        return {"status": "untestable", "questions": len(per_q)}
    m, se = ms([sum(v) / len(v) for v in per_q.values()])
    return {"mean": m, "se": se, "lb": m - Z * se, "ub": m + Z * se, "questions": len(per_q)}


def _by_seed(recs: dict) -> dict:
    out = {}
    for q, rr in recs.items():
        for r in rr:
            got = shares(r)
            if got is not None:
                out[(q, (r.get("config") or {}).get("seed", 0))] = got
    return out


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Token accounting of blackboard cells: shares of waiting, negotiation and "
                                             "work, and budget and margin with every waiting turn removed.")
    ap.add_argument("--spec", required=True, help="grading spec (the format of rci.ladder.grade)")
    ap.add_argument("--task", required=True, help="task name in the spec")
    ap.add_argument("--cells", default=None, help="comma list of cell labels (default: every non-pool cell)")
    ap.add_argument("--pair", action="append", default=[], help='"<X label>|<R label>": paired share difference')
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tspec = json.loads(Path(a.spec).read_text())[a.task]
    pool_recs: dict = {}
    for root, protocol, n in tspec["pool"]:
        for q, rr in records(root, protocol, n).items():
            pool_recs.setdefault(q, []).extend(rr)
    heads = sum(len(_candidates(r)) for rr in pool_recs.values() for r in rr)
    if not heads:
        print("no pool records under " + ", ".join(str(p[0]) for p in tspec["pool"]) + ": run the lines first",
              file=sys.stderr)
        return 1
    c1 = {"completion": sum(float(r["compute"]["completion_tokens"]) for rr in pool_recs.values() for r in rr) / heads,
          "total": sum(float(r["compute"]["total_tokens"]) for rr in pool_recs.values() for r in rr) / heads}
    pool = {q: [m for r in rr for m in members_of(r)] for q, rr in pool_recs.items()}
    t_curves = pass_curves(pool, SIZES, seats=True)
    common = sorted(set.intersection(*(set(c) for c in t_curves.values())))
    pool_keys = {tuple(p) for p in tspec["pool"]}
    wanted = set(a.cells.split(",")) if a.cells else None
    cells = {c["label"]: c for c in tspec["cells"]
             if (c["root"], c["protocol"], c["n"]) not in pool_keys and (wanted is None or c["label"] in wanted)}
    loaded = {lab: records(c["root"], c["protocol"], c["n"]) for lab, c in cells.items()}
    res = {"task": a.task, "c1": c1, "sizes": common, "cells": {}, "pairs": {}}
    for lab, recs in loaded.items():
        res["cells"][lab] = account(recs, c1, t_curves, common)
        row = res["cells"][lab]
        print(f"[ledger] {lab}: wait {row['share_idle'][0]:.3f} coord {row['share_coord'][0]:.3f} "
              f"n_B {row['n_budget_completion_now'][0]:.1f} -> {row['n_budget_completion_free_wait'][0]:.1f} "
              f"margin {row['margin_completion_now'][0]:+.3f} -> {row['margin_completion_free_wait'][0]:+.3f}", flush=True)
    for p in a.pair:
        x, r = p.split("|")
        if x not in loaded or r not in loaded:
            raise SystemExit(f"unknown cell in pair {p!r}")
        sx, sr = _by_seed(loaded[x]), _by_seed(loaded[r])
        res["pairs"][p] = {k: paired_shares(sx, sr, k) for k in ("coord", "wait", "neg", "poll")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ledger_{a.task}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
