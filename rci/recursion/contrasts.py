"""Generation reader: margins over a frozen base pool, steps between generations and control contrasts."""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from rci.ladder.commit import _candidates, vote_stake
from rci.ladder.curves import SIZES, at, members_of, pass_curves
from rci.ladder.records import cost_of
from rci.paths import load_benchmark

DEFAULT_CONTRASTS = (("G0_to_G1", "G1", "G0", "step"), ("G1_to_G2", "G2", "G1", "step"),
                     ("G2_minus_control", "G2", "control", "control"))
MULTIPLIERS = {"step": 1.96, "control": 2.24}
LINEAGE_MULTIPLIERS = {"step": 0.0, "control": 1.0}
TAU_FLOOR = 0.04


def load_spec(path: str) -> dict:
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def load_cell(directory: str, n: int, seeds) -> dict:
    """question -> graded records of <directory>/N<n>/s<seed>/ at the given seeds (all seeds when None)."""
    out: dict = {}
    pats = [os.path.join(directory, f"N{n}", f"s{s}", "*.json*") for s in seeds] if seeds is not None else \
        [os.path.join(directory, f"N{n}", "s*", "*.json*")]
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            if not f.endswith((".json", ".json.gz")) or f.endswith(".chains.jsonl.gz"):
                continue
            with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
                rec = json.load(fh)
            sc = rec.get("score") or {}
            if sc.get("per_agent") and sc.get("score") is not None:
                out.setdefault(os.path.basename(f).split(".json")[0], []).append(rec)
    return out


def pool_entries(spec) -> list:
    """[root, protocol, n] entries of a pool given as a dict, a list of dicts or a list of triples."""
    if isinstance(spec, dict) and "pool" in spec:
        spec = spec["pool"]
    if isinstance(spec, dict):
        spec = [spec]
    return [[e["root"], e["protocol"], int(e["n"])] if isinstance(e, dict) else [e[0], e[1], int(e[2])] for e in spec]


class Pool:
    """Independent members of a frozen model: unit cost c1 and the exact seat-stratified pass@n per question."""

    def __init__(self, entries: list, seeds=None, questions=None, seats: bool = True):
        recs: dict = {}
        for root, protocol, n in entries:
            for q, rr in load_cell(os.path.join(root, protocol), n, seeds).items():
                if questions is None or q in questions:
                    recs.setdefault(q, []).extend(rr)
        pool: dict = {}
        tokens, count = 0.0, 0
        for q, rr in recs.items():
            for r in rr:
                pool.setdefault(q, []).extend(members_of(r))
                tokens += cost_of(r, "completion")
                count += len(_candidates(r))
        self.records = recs
        self.c1 = tokens / max(count, 1)
        self.t = pass_curves(pool, SIZES, seats)
        self.member = {q: sum(m.score for m in v) / len(v) for q, v in pool.items() if v}


@dataclass
class Side:
    """One model's cell read against a pool, with what it takes to repeat the reading on a draw of questions."""
    model: str
    y: dict
    tok: dict
    runs: dict
    t: dict
    c1: float
    pool_n: int

    def on(self, draw: list) -> tuple:
        """(mean Y, mean T(n_B), n_B) on a draw of questions, n_B recomputed from the drawn records."""
        nb = sum(self.tok[q] for q in draw) / sum(self.runs[q] for q in draw) / self.c1
        return sum(self.y[q] for q in draw) / len(draw), sum(at(self.t[q], nb) for q in draw) / len(draw), nb

    @property
    def questions(self) -> list:
        return sorted(self.y)


def side_of(model: str, recs: dict, pool: Pool, questions=None) -> Side | None:
    qs = [q for q in recs if q in pool.t and (questions is None or q in questions)]
    if not qs:
        return None
    common = sorted(set.intersection(*(set(pool.t[q]) for q in qs)))
    if not common:
        return None
    return Side(model, {q: sum(vote_stake(r) for r in recs[q]) / len(recs[q]) for q in qs},
                {q: sum(cost_of(r, "completion") for r in recs[q]) for q in qs}, {q: len(recs[q]) for q in qs},
                {q: {k: v for k, v in pool.t[q].items() if k <= common[-1]} for q in qs}, pool.c1, common[-1])


def draws_of(qs: list, reps: int, salt: str) -> list:
    rng = random.Random(f"contrasts/{salt}")
    return [[qs[rng.randrange(len(qs))] for _ in qs] for _ in range(reps)]


def margin(side: Side, reps: int, salt: str) -> dict:
    """Y - T(n_B) of one model with its bootstrap SE."""
    qs = side.questions
    y, t, nb = side.on(qs)
    reps_ = [side.on(d) for d in draws_of(qs, reps, salt)]
    m = y - t
    se = math.sqrt(sum((a - b - m) ** 2 for a, b, _ in reps_) / max(1, len(reps_) - 1))
    return {"model": side.model, "questions": len(qs), "answer": y, "t_b": t, "n_budget": nb, "pool_n": side.pool_n,
            "capped": nb > side.pool_n, "margin": m, "se": se}


def contrast(pairs: dict, reps: int, salt: str) -> dict:
    """Mean over units of margin(a) - margin(b) with accuracy and cost terms; the bootstrap recomputes n_B."""
    sides = [s for pair in pairs.values() for s in pair if s is not None]
    qs = sorted(set.intersection(*(set(s.y) for s in sides))) if sides else []
    if not qs:
        return {"mean": None, "se": None, "n": 0}

    def stat(draw: list) -> dict:
        out = {}
        for u, (a, b) in pairs.items():
            ya, ta, na = a.on(draw)
            yb, tb, nb = b.on(draw) if b is not None else (0.0, 0.0, None)
            out[u] = (ya - yb, ta - tb, na, nb)
        return out

    point = stat(qs)
    boot = [stat(d) for d in draws_of(qs, reps, salt)]

    def est(f) -> dict:
        x = f(point)
        return {"mean": x, "se": math.sqrt(sum((f(d) - x) ** 2 for d in boot) / max(1, len(boot) - 1))}

    def avg(i: int, units=None):
        return lambda s: sum(s[u][i] for u in (units or s)) / len(units or s)

    out = {**est(lambda s: avg(0)(s) - avg(1)(s)), "n": len(qs), "units": len(pairs),
           "accuracy": est(avg(0)), "cost": est(lambda s: -avg(1)(s))}
    lineages = sorted({u.split("/")[0] for u in pairs})
    out["lineages"] = {}
    for lin in lineages:
        us = [u for u in pairs if u.split("/")[0] == lin]
        out["lineages"][lin] = est(lambda s, us=us: avg(0, us)(s) - avg(1, us)(s))
    out["units_detail"] = {u: est(lambda s, u=u: s[u][0] - s[u][1]) for u in pairs}
    out["capped"] = any((a.on(qs)[2] > a.pool_n) or (b is not None and b.on(qs)[2] > b.pool_n)
                        for a, b in pairs.values())
    return out


def tau_hat(diffs: list) -> float | None:
    """Training spread from pairwise differences (d, se) of models trained alike: max(0, mean(d^2 - se^2) / 2)."""
    if not diffs:
        return None
    return math.sqrt(max(0.0, sum(d * d - se * se for d, se in diffs) / len(diffs) / 2))


def se_total(se_q: float, tau: float, c: float) -> float:
    """Standard error that includes training randomness: sqrt(SE_q^2 + c tau^2)."""
    return math.sqrt(se_q * se_q + c * tau * tau)


def holm(ps: list) -> list:
    idx = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: ps[i])
    m, out, run = len(idx), [None] * len(ps), 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (m - r) * ps[i]))
        out[i] = run
    return out


def p_one(mean: float, se: float) -> float:
    if not se:
        return 0.0 if mean > 0 else 1.0
    return 0.5 * math.erfc(mean / se / math.sqrt(2))


def lineup(models) -> dict:
    """unit id ('l<i>' or 'l<i>/t<j>') -> served name, for a flat list (one per lineage) or nested lists (seeds)."""
    if isinstance(models, str):
        models = [models]
    out = {}
    for i, m in enumerate(models):
        if isinstance(m, (list, tuple)):
            for j, x in enumerate(m):
                out[f"l{i}/t{j}"] = x
        else:
            out[f"l{i}"] = m
    return out


def pair_units(a: dict, b: dict) -> dict:
    """Pair the units of two generations: same unit, or a's unit with b's lineage, or b's single model."""
    out = {}
    for u, ma in a.items():
        lin = u.split("/")[0]
        if u in b:
            mb = b[u]
        elif lin in b:
            mb = b[lin]
        elif len(b) == 1:
            mb = next(iter(b.values()))
        else:
            continue
        out[u] = (ma, mb)
    return out


def tau_of(models: dict, sides: dict, reps: int, salt: str) -> float | None:
    """tau-hat from the pairwise margin differences of the seeds of each lineage."""
    diffs = []
    by_lin: dict = {}
    for u, m in models.items():
        if "/" in u:
            by_lin.setdefault(u.split("/")[0], []).append(m)
    for lin, ms in by_lin.items():
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                c = contrast({lin: (sides[ms[i]], sides[ms[j]])}, reps, f"{salt}/tau/{ms[i]}/{ms[j]}")
                if c["mean"] is not None:
                    diffs.append((c["mean"], c["se"]))
    return tau_hat(diffs)


def holds(c: dict, z: float, lz: float, tau: float | None, c_pooled: float, c_lineage: float) -> dict:
    se = c["se"] if tau is None else se_total(c["se"], tau, c_pooled)
    lin_ok = all(v["mean"] - lz * (v["se"] if tau is None else se_total(v["se"], tau, c_lineage)) > 0
                 for v in c["lineages"].values())
    return {"z": z, "lineage_z": lz, "se_used": se, "pooled": c["mean"] - z * se > 0, "each_lineage": lin_ok,
            "not_capped": not c["capped"], "holds": c["mean"] - z * se > 0 and lin_ok and not c["capped"]}


def first_turn_report(recs: dict, bench: dict | None) -> dict:
    """Share of members whose first message writes every private fact, and the mean length of the first call."""
    from rci.recursion.turns import first_turn_stats

    full, total, tokens = 0, 0, []
    for q, rr in recs.items():
        for r in rr:
            for a in r.get("agents") or []:
                calls = [c for c in a.get("calls") or [] if isinstance(c, dict)]
                if calls and calls[0].get("completion_tokens") is not None:
                    tokens.append(float(calls[0]["completion_tokens"]))
            if bench is None or q not in bench or not r.get("transcripts"):
                continue
            st = first_turn_stats(r, bench[q])
            kept = [m for m in st["members"] if m["left_out"] is None]
            full += sum(m["all"] for m in kept)
            total += len(kept)
    return {"members_measured": total, "all_facts_share": full / total if total else None,
            "first_call_tokens": sum(tokens) / len(tokens) if tokens else None}


def questions_of(spec: dict) -> set | None:
    if spec.get("questions"):
        return set(json.loads(Path(spec["questions"]).read_text()))
    if spec.get("heldout"):
        rows = load_benchmark(spec["heldout"])
        return {f"hb_{int(r['id']):02d}" for r in rows}
    return None


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read the label-free generations: each model's margin over pass@n of a "
                                             "frozen base pool, and the contrasts between generations averaged over "
                                             "lineages.")
    ap.add_argument("spec", help="YAML or JSON spec (see the repository README)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-turns", action="store_true", help="also report first-turn coverage and first-call length")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    reps = int(spec.get("draws", 2000))
    qset = questions_of(spec)
    pool = Pool(pool_entries(spec["base_pool"]), spec.get("pool_seeds"), qset)
    n, seeds = int(spec.get("n", 4)), spec.get("seeds")
    gens = {g: lineup(m) for g, m in spec["generations"].items()}
    names = sorted({m for units in gens.values() for m in units.values()})
    cells = {m: load_cell(os.path.join(spec["eval_root"].format(model=m), spec["protocol"]), n, seeds) for m in names}
    sides = {m: side_of(m, cells[m], pool, qset) for m in names}
    missing = [m for m, s in sides.items() if s is None]
    if missing:
        raise SystemExit(f"no graded records for {missing}")
    res = {"pool": {"c1": pool.c1, "questions": len(pool.t)}, "draws": reps, "models": {}, "generations": {},
           "contrasts": {}}
    for m in names:
        res["models"][m] = margin(sides[m], reps, f"margin/{m}")
    for g, units in gens.items():
        vals = [res["models"][m]["margin"] for m in units.values()]
        res["generations"][g] = {"margin": sum(vals) / len(vals), "models": dict(units)}
    mult = {**MULTIPLIERS, **(spec.get("multipliers") or {})}
    lmult = {**LINEAGE_MULTIPLIERS, **(spec.get("lineage_multipliers") or {})}
    tau_floor = float(spec.get("tau_floor", TAU_FLOOR))
    rows = [tuple(c) for c in spec.get("contrasts") or DEFAULT_CONTRASTS]
    ps = []
    for name, ga, gb, kind in rows:
        if ga not in gens or gb not in gens:
            continue
        units = pair_units(gens[ga], gens[gb])
        c = contrast({u: (sides[x], sides[y]) for u, (x, y) in units.items()}, reps, f"contrast/{name}")
        seeded = any("/" in u for u in units)
        tau = None
        if seeded:
            t_hat = tau_of(gens[ga], sides, reps, name)
            tau = max(tau_floor, t_hat or 0.0)
            c["tau_hat"], c["tau"] = t_hat, tau
            both = any("/" in u for u in gens[gb])
            k = 2 if both else 1
            n_units = len(units)
            n_lin = max(1, len({u.split("/")[0] for u in units}))
            c["se_total"] = se_total(c["se"], tau, k / n_units)
            c["rule"] = holds(c, mult.get(kind, 1.96), lmult.get(kind, 0.0), tau, k / n_units, k * n_lin / n_units)
        else:
            c["rule"] = holds(c, mult.get(kind, 1.96), lmult.get(kind, 0.0), None, 0.0, 0.0)
        c["kind"], c["a"], c["b"] = kind, ga, gb
        res["contrasts"][name] = c
        ps.append(p_one(c["mean"], c["rule"]["se_used"]) if c["mean"] is not None else None)
    for (name, *_), q in zip([r for r in rows if r[0] in res["contrasts"]], holm(ps)):
        res["contrasts"][name]["holm_p"] = q
    if a.first_turns:
        bench = None
        if spec.get("heldout"):
            bench = {f"hb_{int(r['id']):02d}": r for r in load_benchmark(spec["heldout"])}
        res["first_turns"] = {m: first_turn_report(cells[m], bench) for m in names}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "contrasts.json").write_text(json.dumps(res, indent=1))
    lines = ["| model | q | n_B | Y | T(n_B) | margin |", "|---|---|---|---|---|---|"]
    for m, r in res["models"].items():
        cap = " (capped)" if r["capped"] else ""
        lines.append(f"| {m} | {r['questions']} | {r['n_budget']:.1f}{cap} | {r['answer']:.3f} | {r['t_b']:.3f} | "
                     f"{r['margin']:+.3f} +- {r['se']:.3f} |")
    lines += ["", "| contrast | mean | SE | accuracy | cost | holds |", "|---|---|---|---|---|---|"]
    for name, c in res["contrasts"].items():
        if c["mean"] is None:
            lines.append(f"| {name} | - | - | - | - | - |")
            continue
        lines.append(f"| {name} | {c['mean']:+.3f} | {c['rule']['se_used']:.3f} | {c['accuracy']['mean']:+.3f} | "
                     f"{c['cost']['mean']:+.3f} | {'yes' if c['rule']['holds'] else 'no'} |")
    (out / "contrasts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
