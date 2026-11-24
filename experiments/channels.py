from __future__ import annotations

import sys

from _common import grade_spec, load, parser, run, write_json

from rci import paths


def spec(exp: dict) -> dict:
    out = {}
    for name, t in exp["tasks"].items():
        src = grade_spec(load(t["experiment"]), [t["model"]])
        out[name] = src[f"{t['grade_task']}_{t['model']}"]
    return out


def main(argv=None) -> int:
    ap, sub, common = parser("Accuracy and cost terms (Figure 7, Table channels a) and token accounting (Table channels b).")
    sub.add_parser("channels", parents=[common], help="accuracy / cost decomposition of every pair")
    sub.add_parser("ledger", parents=[common], help="token accounting and the replay without waiting")
    a = ap.parse_args(argv)
    exp = load("channels")
    if a.cmd in ("lines", "cells"):
        print("channels reuses the records of experiments/ladder.py and experiments/emergence.py", file=sys.stderr)
        return 0
    s = spec(exp)
    path = write_json(paths.results_dir("channels", "spec.json"), s)
    out = str(paths.results_dir("channels"))
    rc = 0
    if a.cmd in ("channels", "grade"):
        for task in s:
            pairs = [[x, r] for t, x, r in exp["pairs"] if t == task]
            if not pairs:
                continue
            pf = write_json(paths.results_dir("channels", f"pairs_{task}.json"), pairs)
            rc |= run([sys.executable, "-m", "rci.ladder.channels", "--spec", str(path), "--task", task, "--pairs",
                       str(pf), "--out", out], a.dry_run)
    if a.cmd in ("ledger", "grade"):
        for task, labels in exp["ledger"].items():
            rc |= run([sys.executable, "-m", "rci.ladder.ledger", "--spec", str(path), "--task", task, "--cells",
                       ",".join(labels), "--out", out], a.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
