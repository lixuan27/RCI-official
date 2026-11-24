"""Accuracy and cost terms of a protocol against a reference on the same pool (Eq. 1 of the paper)."""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path

from rci.ladder import grade
from rci.ladder.commit import _candidates
from rci.ladder.curves import at, classify, members_of
from rci.ladder.records import answer_of, cost_of, records

COSTS = ("completion", "total")
Z = 1.96


def table_flags(cost: str, seed: int, draws: int) -> list:
    """The grading options the decomposition is defined under."""
    return ["--stratify-seats", "--seat-split", "balanced", "--strict-pool", "--bh-family", "census", "--task-rng",
            "--cost", cost, "--seed", str(seed), "--draws", str(draws)]


def run_table(spec_path: str, task: str, out: Path, cost: str, seed: int, draws: int, vote_free: bool) -> dict:
    """Run rci.ladder.grade on one task and return its rows by cell label."""
    with contextlib.redirect_stdout(sys.stderr):
        grade.main([spec_path, "--out", str(out), "--only", task, *table_flags(cost, seed, draws)])
    tag = cost + "_strat_bal_strict" + ("_votefree" if vote_free else "") + "_census_taskrng"
    rows = json.loads((out / f"{task}_{tag}.json").read_text())["rows"]
    return {r["cell"]: r for r in rows}


def pool_and_curves(tspec: dict, name: str, seed: int, draws: int):
    """Pool, unit cost per cost kind and the per-question curves, drawn as rci.ladder.grade draws them."""
    rng = random.Random(f"{seed}:{name}")
    free = bool(tspec.get("vote_free"))
    cells = {(root, protocol, n): records(root, protocol, n) for root, protocol, n in tspec["pool"]}
    pool: dict = {}
    tokens = {c: 0.0 for c in COSTS}
    count = 0
    for recs in cells.values():
        for q, rr in recs.items():
            for r in rr:
                pool.setdefault(q, []).extend(members_of(r))
                for c in COSTS:
                    tokens[c] += cost_of(r, c)
                count += len(_candidates(r))
    c1 = {c: tokens[c] / max(count, 1) for c in COSTS}
    if tspec.get("min_pool"):
        pool = {q: v for q, v in pool.items() if len(v) >= tspec["min_pool"]}
    cs = grade.curves(tspec, pool, rng, draws, True, free, True)
    return pool, c1, cs, free


def cell_pq(recs: dict, cs: dict, c1: float, cost: str, free: bool) -> dict | None:
    """Per-question Y, the cell's mean budget n_B and the sizes every question's curve has."""
    ys, spend = {}, []
    for q, rr in recs.items():
        if q not in cs:
            continue
        got = [(answer_of(r, False, None, free), r) for r in rr]
        got = [(y, r) for y, r in got if y is not None]
        if not got:
            continue
        ys[q] = sum(y for y, _ in got) / len(got)
        spend += [cost_of(r, cost) for _, r in got]
    if not ys:
        return None
    common = sorted(set.intersection(*(set(cs[q]) for q in ys)))
    return {"y": ys, "n_b": sum(spend) / len(spend) / c1, "common": common}


def e_q(pq: dict, cs: dict, ref: int, n: float) -> dict:
    """Per question Y - curve(n), with ref 0 the vote V and ref 1 pass@n T."""
    return {q: y - at({k: cs[q][k][ref] for k in pq["common"]}, n) for q, y in pq["y"].items()}


def contrast(d: dict) -> dict:
    m, se = grade.ms(list(d.values()))
    return {"mean": m, "se": se, "class": classify(m, se), "questions": len(d)}


def passes(c: dict) -> bool:
    return c["mean"] - Z * c["se"] > 0


def paired(a: dict, b: dict) -> dict:
    qs = sorted(set(a) & set(b))
    return contrast({q: a[q] - b[q] for q in qs})


def t_at(pq: dict, cs: dict, n: float, common: list) -> dict:
    return {q: at({k: cs[q][k][1] for k in common}, n) for q in pq["y"]}


def split(px: dict, pr: dict, cs: dict) -> dict:
    """[Y_X - T(n_X)] - [Y_R - T(n_R)] = (Y_X - Y_R) + [T(n_R) - T(n_X)], and the margin of X at R's budget."""
    qs = sorted(set(px["y"]) & set(pr["y"]))
    common = sorted(set(px["common"]) & set(pr["common"]))
    x = {**px, "y": {q: px["y"][q] for q in qs}}
    r = {**pr, "y": {q: pr["y"][q] for q in qs}}
    tx_own, tr_own = t_at(x, cs, x["n_b"], common), t_at(r, cs, r["n_b"], common)
    tx_at_r = t_at(x, cs, r["n_b"], common)
    pool_n = common[-1]
    out = {"questions": len(qs), "n_b_x": x["n_b"], "n_b_r": r["n_b"], "pool_n": pool_n,
           "y_x": sum(x["y"].values()) / len(qs), "y_r": sum(r["y"].values()) / len(qs),
           "margin_x": paired(x["y"], tx_own), "margin_r": paired(r["y"], tr_own),
           "x_at_r_budget": paired(x["y"], tx_at_r), "accuracy": paired(x["y"], r["y"]),
           "cost": paired(tr_own, tx_own)}
    out["margin_change"] = paired({q: x["y"][q] - tx_own[q] for q in qs}, {q: r["y"][q] - tr_own[q] for q in qs})
    if x["n_b"] > pool_n or r["n_b"] > pool_n:
        out["via"] = "capped"
    elif not passes(out["margin_x"]):
        out["via"] = "none"
    elif passes(out["x_at_r_budget"]):
        out["via"] = "accuracy"
    else:
        out["via"] = "accuracy_and_cost"
    return out


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Split the margin change of a protocol against a reference into an "
                                             "accuracy term and a cost term, on one task of a grading spec.")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--pair", action="append", default=[], help='"<protocol label>|<reference label>"')
    ap.add_argument("--pairs", default=None, help="JSON list of [protocol label, reference label] pairs, or a "
                                                  "mapping from task name to such a list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--costs", default=",".join(COSTS))
    a = ap.parse_args(argv)
    costs = tuple(c for c in a.costs.split(",") if c)
    tspec = json.loads(Path(a.spec).read_text())[a.task]
    pairs = [tuple(p.split("|")) for p in a.pair]
    if a.pairs:
        got = json.loads(Path(a.pairs).read_text())
        pairs += [tuple(p) for p in (got.get(a.task, []) if isinstance(got, dict) else got)]
    if not pairs:
        raise SystemExit(f"{a.task}: no pairs")
    by_label = {c["label"]: c for c in tspec["cells"]}
    pool_keys = {tuple(p) for p in tspec["pool"]}
    for p in pairs:
        if len(p) != 2:
            raise SystemExit(f"bad pair {p}")
        for lab in p:
            if lab not in by_label:
                raise SystemExit(f"{a.task}: no cell {lab!r}")
            c = by_label[lab]
            if (c["root"], c["protocol"], c["n"]) in pool_keys:
                raise SystemExit(f"{a.task}: {lab!r} is a pool cell")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {c: run_table(a.spec, a.task, out, c, a.seed, a.draws, bool(tspec.get("vote_free"))) for c in costs}
    _, c1, cs, free = pool_and_curves(tspec, a.task, a.seed, a.draws)
    pq = {c: {} for c in costs}
    checks = {}
    for lab in sorted({lab for p in pairs for lab in p}):
        cell = by_label[lab]
        recs = records(cell["root"], cell["protocol"], cell["n"])
        for c in costs:
            got = cell_pq(recs, cs, c1[c], c, free)
            if got is None:
                continue
            pq[c][lab] = got
            row = tables[c].get(lab)
            y = sum(got["y"].values()) / len(got["y"])
            et = contrast(e_q(got, cs, 1, got["n_b"]))["mean"]
            checks[f"{lab}/{c}"] = bool(row is not None and abs(y - row["answer"]) < 1e-9
                                        and abs(got["n_b"] - row["n_budget"]) < 1e-6
                                        and abs(et - row["e_truth"][0]) < 1e-9)
    res = {"task": a.task, "options": table_flags("<cost>", a.seed, a.draws), "c1": c1, "checks": checks, "pairs": {}}
    for x, r in pairs:
        key = f"{x} || {r}"
        res["pairs"][key] = {}
        for c in costs:
            if x not in pq[c] or r not in pq[c]:
                res["pairs"][key][c] = {"missing": True}
                continue
            s = split(pq[c][x], pq[c][r], cs)
            s["level_x"] = tables[c][x]["level"] if x in tables[c] else None
            s["level_r"] = tables[c][r]["level"] if r in tables[c] else None
            s["table_check"] = bool(checks.get(f"{x}/{c}") and checks.get(f"{r}/{c}"))
            if not s["table_check"]:
                s["via"] = "void: table mismatch"
            res["pairs"][key][c] = s
    path = out / f"channels_{a.task}.json"
    path.write_text(json.dumps(res, indent=1))
    for key, v in res["pairs"].items():
        for c, s in v.items():
            if s.get("missing"):
                print(f"[channels] {key} {c}: missing", flush=True)
                continue

            def f(d):
                return f"{d['mean']:+.3f}+-{d['se']:.3f}"

            print(f"[channels] {c} {key}: n_B {s['n_b_x']:.1f}/{s['n_b_r']:.1f} margin {f(s['margin_x'])} vs "
                  f"{f(s['margin_r'])} | accuracy {f(s['accuracy'])} cost {f(s['cost'])} | at ref. "
                  f"{f(s['x_at_r_budget'])} -> {s['via']}", flush=True)
    return res


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
