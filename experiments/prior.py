from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import cells, dispatch, grade_spec, load, parser, protocol_dir, run, shell, write_json

from rci import paths

JUDGE_MODULES = {"browsecomp": "rci.tasks.browsecomp.grade", "mmlongdoc": "rci.tasks.mmlongdoc.grade"}


def review_dir(cell: dict, reviewer: str) -> str:
    return str(Path(cell["root"]) / f"review_{reviewer}_n{cell['n_agents']}")


def by_name(exp: dict) -> dict:
    return {c["name"]: c for c in cells(exp)}


def spec(exp: dict) -> dict:
    got = {name.rsplit("_q9", 1)[0]: t for name, t in grade_spec(exp).items()}
    named = by_name(exp)
    for task, t in exp["grade"]["tasks"].items():
        for ref in t.get("reviews", []):
            c = named[ref]
            for key, r in exp["reviewers"].items():
                row = {"label": f"{c['label']}+{r['label']}", "root": c["root"], "protocol": protocol_dir(c),
                       "n": c["n_agents"], "review": review_dir(c, key)}
                if r.get("review_scale"):
                    row["review_scale"] = r["review_scale"]
                got[task]["cells"].append(row)
    return got


def reviews(exp: dict, a) -> int:
    named = by_name(exp)
    rc = 0
    for item in exp["reviews"]:
        c = named[item["cell"]]
        for key, r in exp["reviewers"].items():
            if a.reviewer and key != a.reviewer:
                continue
            argv = [sys.executable, "-m", "rci.ladder.reviewer", "replay", "--root", c["root"], "--protocol",
                    protocol_dir(c), "--n", str(c["n_agents"]), "--task", item["task"], "--task-kwargs",
                    json.dumps(item["task_kwargs"]), "--mode", item["mode"], "--model", r["served"],
                    "--out", review_dir(c, key)]
            argv += ["--base-urls", a.base_urls] if a.base_urls else []
            rc |= run(argv, a.dry_run)
    return rc


def judge(exp: dict, a) -> int:
    roots = {}
    for c in cells(exp):
        if c["task"] in JUDGE_MODULES:
            roots.setdefault(c["task"], set()).add(c["root"])
    rc = 0
    for task, rs in roots.items():
        argv = [sys.executable, "-m", JUDGE_MODULES[task], *sorted(rs), "--base-url", a.base_url or "$RCI_JUDGE_URL",
                "--model", exp["judge"]]
        if a.dry_run or not a.base_url:
            print(shell(argv))
        else:
            rc |= run(argv, False)
    return rc


def main(argv=None) -> int:
    ap, sub, common = parser("Prior multi-agent methods against the equal-budget vote (Figure 3c, Table prior).")
    j = sub.add_parser("judge", parents=[common], help="grade BrowseComp-Plus and MMLongBench-Doc with the judge")
    j.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint serving the judge model")
    r = sub.add_parser("reviews", parents=[common], help="replay the 9B and 27B reviewers on blackboard records")
    r.add_argument("--reviewer", default=None, help="q9 or q27 (default both)")
    r.add_argument("--base-urls", default=None)
    t = sub.add_parser("table", parents=[common], help="write the prior-methods table from the graded census")
    t.add_argument("--unit", default="completion")
    a = ap.parse_args(argv)
    exp = load("prior")
    if a.cmd == "grade":
        path = write_json(paths.results_dir("prior", "grade_spec.json"), spec(exp))
        out = str(paths.results_dir("prior", "census_completion"))
        return run([sys.executable, "-m", "rci.ladder.grade", str(path), "--out", out, "--cost", "completion"]
                   + [str(x) for x in exp["grade"]["options"]], a.dry_run)
    got = dispatch(exp, a)
    if got is not None:
        return got
    if a.cmd == "judge":
        return judge(exp, a)
    if a.cmd == "reviews":
        return reviews(exp, a)
    root = str(paths.results_dir("prior"))
    rc = 0
    for out in ("table.md", "table.tex"):
        rc |= run([sys.executable, "-m", "rci.ladder.prior_table", root, "--unit", a.unit, "--out",
                   str(paths.results_dir("prior", out))], a.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
