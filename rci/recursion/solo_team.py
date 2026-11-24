"""Solo accuracy with every fact against the team's gain under split facts, step by step."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from rci.ladder.commit import vote_stake
from rci.recursion.contrasts import draws_of, lineup, load_cell, load_spec

SOLVED = 0.75
BOUND = 0.03


def per_question(recs: dict) -> dict:
    """Committed score per question, averaged over its records."""
    return {q: sum(vote_stake(r) for r in rr) / len(rr) for q, rr in recs.items() if rr}


def steps(spec: dict) -> list:
    """(name, parent units, child units) of every step, pairing child lineages with their parents."""
    gens = {g: lineup(m) for g, m in spec["generations"].items()}
    out = []
    for name, child, parent in [tuple(s) for s in spec.get("steps") or (("G0_to_G1", "G1", "G0"),
                                                                          ("G1_to_G2", "G2", "G1"))]:
        if child not in gens or parent not in gens:
            continue
        pairs = {}
        for u, mc in gens[child].items():
            lin = u.split("/")[0]
            mp = gens[parent].get(u) or gens[parent].get(lin) or (next(iter(gens[parent].values()))
                                                                   if len(gens[parent]) == 1 else None)
            if mp is not None:
                pairs[u] = (mp, mc)
        out.append((name, pairs))
    return out


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="For each step between generations: the change of one member's accuracy "
                                             "given every fact (individual) and the change of the group's accuracy "
                                             "under split facts on the tasks both generations solve alone (team).")
    ap.add_argument("spec", help="generations, solo_root and team_root templates with {model}, protocols, seeds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    reps = int(spec.get("draws", 2000))
    solved = float(spec.get("solved", SOLVED))
    bound = float(spec.get("bound", BOUND))
    gens = {g: lineup(m) for g, m in spec["generations"].items()}
    names = sorted({m for units in gens.values() for m in units.values()})
    solo, team = {}, {}
    for m in names:
        solo[m] = per_question(load_cell(os.path.join(spec["solo_root"].format(model=m), spec["solo_protocol"]),
                                         int(spec.get("solo_n", 1)), spec.get("solo_seeds")))
        team[m] = per_question(load_cell(os.path.join(spec["team_root"].format(model=m), spec["team_protocol"]),
                                         int(spec.get("team_n", 4)), spec.get("team_seeds")))
    res = {"solved_at": solved, "equivalence_bound": bound, "models": {}, "steps": {}}
    for m in names:
        res["models"][m] = {"solo": sum(solo[m].values()) / max(1, len(solo[m])),
                            "team": sum(team[m].values()) / max(1, len(team[m]))}
    for name, pairs in steps(spec):
        qs = sorted(set.intersection(*(set(solo[p]) & set(solo[c]) & set(team[p]) & set(team[c])
                                       for p, c in pairs.values())))
        if not qs:
            continue

        def stat(draw):
            ind = sum(solo[c][q] - solo[p][q] for p, c in pairs.values() for q in draw) / (len(pairs) * len(draw))
            parts = []
            for p, c in pairs.values():
                both = [q for q in draw if solo[p][q] >= solved and solo[c][q] >= solved]
                if both:
                    parts.append(sum(team[c][q] - team[p][q] for q in both) / len(both))
            return ind, (sum(parts) / len(parts) if parts else float("nan"))

        point = stat(qs)
        boot = [stat(d) for d in draws_of(qs, reps, f"solo_team/{name}")]

        def se(i):
            xs = [b[i] for b in boot if not math.isnan(b[i])]
            return math.sqrt(sum((x - point[i]) ** 2 for x in xs) / max(1, len(xs) - 1))

        ind_se, team_se = se(0), se(1)
        res["steps"][name] = {
            "questions": len(qs),
            "solo_parent": sum(solo[p][q] for p, _ in pairs.values() for q in qs) / (len(pairs) * len(qs)),
            "solo_child": sum(solo[c][q] for _, c in pairs.values() for q in qs) / (len(pairs) * len(qs)),
            "individual": {"mean": point[0], "se": ind_se,
                           "interval_90": [point[0] - 1.645 * ind_se, point[0] + 1.645 * ind_se],
                           "within_bound": abs(point[0]) + 1.645 * ind_se < bound},
            "team": {"mean": point[1], "se": team_se,
                     "solved_by_both": {u: sum(1 for q in qs if solo[p][q] >= solved and solo[c][q] >= solved)
                                        for u, (p, c) in pairs.items()}}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "solo_team.json").write_text(json.dumps(res, indent=1))
    lines = ["| step | solo | individual | team |", "|---|---|---|---|"]
    for name, r in res["steps"].items():
        lines.append(f"| {name} | {r['solo_parent']:.3f} -> {r['solo_child']:.3f} | {r['individual']['mean']:+.3f} +- "
                     f"{r['individual']['se']:.3f} | {r['team']['mean']:+.3f} +- {r['team']['se']:.3f} |")
    (out / "solo_team.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
