"""Table 3, Figures 5 and 6: scarce attention, self-written prompts, budget caps and the structure groups form."""
from __future__ import annotations

import json
import os
import sys

from _common import cells, dispatch, expand, keys, load, models, parser, protocol_dir, run, write_json

from rci import paths


def structure(exp: dict, a) -> int:
    rc = 0
    for block, b in exp["structure"].items():
        if block == "pairs":
            continue
        for key in b["models"]:
            if keys(a) and key not in keys(a):
                continue
            got = [c for c in cells(exp, [key]) if c["name"] in b["cells"]]
            spec = {f"{block}_{key}": {"task": "hiddenbench", "pool": [], "cells": [
                {"label": c["label"], "name": c["name"], "root": c["root"], "protocol": protocol_dir(c),
                 "n": c["n_agents"], "protocol_kwargs": c["protocol_kwargs"]} for c in got],
                "pairs": [p for p in exp["structure"].get("pairs", []) if all(x in b["cells"] for x in p)]}}
            path = write_json(paths.results_dir("emergence", "structure", f"{block}_{key}.json"), spec)
            argv = [sys.executable, "-m", "rci.organize.structure", "read", "--spec", str(path), "--task",
                    f"{block}_{key}", "--out", str(paths.results_dir("emergence", "structure")), "--null", str(a.null),
                    "--hubs"]
            rc |= run(argv, a.dry_run)
    return rc


def prompts(exp: dict, a) -> int:
    rc = 0
    ms = models(exp)
    data = str(paths.data_dir("splitbench", "train"))
    urls = a.base_urls or os.environ.get("RCI_BASE_URLS")
    if not urls and not a.dry_run:
        print("set RCI_BASE_URLS or pass --base-urls", file=sys.stderr)
        return 2
    for name, p in exp["self_prompt"].items():
        if a.which and name not in a.which.split(","):
            continue
        p = expand(p)
        for seed in p.get("seeds", [0]):
            tk = {"root": data, **p["task_kwargs"]}
            argv = [sys.executable, "-m", "rci.organize.self_prompt", "hiddenbench", "--kind", p["kind"],
                    "--base-urls", urls or "$RCI_BASE_URLS", "--model", ms[p["model"]]["served"],
                    "--task-kwargs", json.dumps(tk), "--protocol-kwargs", json.dumps(p["protocol_kwargs"]),
                    "--seed", str(seed), "--out", p["out"].format(seed=seed)]
            if p["kind"] == "evolved":
                argv += ["--generations", str(p["generations"]), "--per-generation", str(p["per_generation"])]
            argv += [str(x) for x in p.get("args", [])]
            rc |= run(argv, a.dry_run)
    return rc


def main(argv=None) -> int:
    ap, sub, common = parser("Emergence under scarce attention, Table 3 and Figures 5-6.")
    s = sub.add_parser("structure", parents=[common], help="structure reader and null policy")
    s.add_argument("--null", type=int, default=10000, help="null-policy simulations per question")
    p = sub.add_parser("prompts", parents=[common], help="write self-written prompts (rules or evolved)")
    p.add_argument("--which", default=None, help="comma list of entries of self_prompt in the YAML")
    p.add_argument("--base-urls", default=None)
    a = ap.parse_args(argv)
    exp = load("emergence")
    got = dispatch(exp, a)
    if got is not None:
        return got
    if a.cmd == "structure":
        return structure(exp, a)
    return prompts(exp, a)


if __name__ == "__main__":
    sys.exit(main())
