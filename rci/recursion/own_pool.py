"""Each generation graded against pass@n of its own members' independent runs, beside the frozen base pool."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from rci.ladder.curves import at
from rci.recursion.contrasts import (Pool, contrast, draws_of, lineup, load_cell, load_spec, margin, pair_units,
                                     pool_entries, questions_of, side_of)


def pool_spec(template, model: str) -> list:
    """The own pool's [root, protocol, n] entries, with {model} filled in."""
    entries = pool_entries(template)
    return [[root.format(model=model), protocol, n] for root, protocol, n in entries]


def member_contrast(a: dict, b: dict, reps: int, salt: str) -> dict:
    """Mean single-member accuracy of pool a minus pool b, paired by question, with a bootstrap SE."""
    qs = sorted(set(a) & set(b))
    if not qs:
        return {"mean": None, "se": None, "n": 0}

    def f(draw):
        return sum(a[q] - b[q] for q in draw) / len(draw)

    m = f(qs)
    boot = [f(d) for d in draws_of(qs, reps, salt)]
    return {"mean": m, "se": math.sqrt(sum((x - m) ** 2 for x in boot) / max(1, len(boot) - 1)), "n": len(qs)}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grade each generation against a pool of its own members working "
                                             "independently: margins, steps under the own pools, and the change of a "
                                             "member working alone.")
    ap.add_argument("spec", help="the contrasts spec plus own_pool (a pool template with {model}) and own_pool_seeds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    reps = int(spec.get("draws", 2000))
    qset = questions_of(spec)
    n, seeds = int(spec.get("n", 4)), spec.get("seeds")
    base = Pool(pool_entries(spec["base_pool"]), spec.get("pool_seeds"), qset)
    gens = {g: lineup(m) for g, m in spec["generations"].items()}
    names = sorted({m for units in gens.values() for m in units.values()})
    own = {m: Pool(pool_spec(spec["own_pool"], m), spec.get("own_pool_seeds"), qset) for m in names}
    cells = {m: load_cell(os.path.join(spec["eval_root"].format(model=m), spec["protocol"]), n, seeds) for m in names}
    own_side = {m: side_of(m, cells[m], own[m], qset) for m in names}
    base_side = {m: side_of(m, cells[m], base, qset) for m in names}
    res = {"models": {}, "steps": {}}
    for m in names:
        if own_side[m] is None:
            res["models"][m] = {"missing": True}
            continue
        mo = margin(own_side[m], reps, f"own/{m}")
        mb = margin(base_side[m], reps, f"base/{m}") if base_side[m] is not None else None
        qs = own_side[m].questions
        nb = mo["n_budget"]
        gap = [at(own[m].t[q], nb) - at(base.t[q], nb) for q in qs if q in base.t]
        res["models"][m] = {"own": mo, "base": mb,
                            "own_minus_base_pass": sum(gap) / len(gap) if gap else None,
                            "member_alone": sum(own[m].member[q] for q in qs if q in own[m].member)
                            / max(1, sum(q in own[m].member for q in qs))}
    for name, ga, gb, _ in [tuple(c) for c in spec.get("contrasts") or
                            (("G0_to_G1", "G1", "G0", "step"), ("G1_to_G2", "G2", "G1", "step"))]:
        if ga not in gens or gb not in gens:
            continue
        units = pair_units(gens[ga], gens[gb])
        if any(own_side[x] is None or own_side[y] is None for x, y in units.values()):
            continue
        c = contrast({u: (own_side[x], own_side[y]) for u, (x, y) in units.items()}, reps, f"own/{name}")
        alone = [member_contrast(own[x].member, own[y].member, reps, f"alone/{name}/{u}") for u, (x, y) in units.items()]
        res["steps"][name] = {"own_pools": c,
                              "member_alone": {"mean": sum(x["mean"] for x in alone if x["mean"] is not None)
                                               / max(1, sum(x["mean"] is not None for x in alone)),
                                               "units": alone}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "own_pool.json").write_text(json.dumps(res, indent=1))
    lines = ["| model | own margin | base margin | own minus base pass@n | member alone |", "|---|---|---|---|---|"]
    for m, r in res["models"].items():
        if r.get("missing"):
            continue
        b = "-" if r["base"] is None else f"{r['base']['margin']:+.3f}"
        g = "-" if r["own_minus_base_pass"] is None else f"{r['own_minus_base_pass']:+.3f}"
        lines.append(f"| {m} | {r['own']['margin']:+.3f} +- {r['own']['se']:.3f} | {b} | {g} | {r['member_alone']:.3f} |")
    (out / "own_pool.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
