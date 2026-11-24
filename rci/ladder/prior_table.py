"""Table of prior multi-agent methods against the equal-budget vote and pass@n of independent agents."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

UNITS = ("completion", "total", "agents")
TASKS = (("gpqa", "GPQA-D"), ("browsecomp", "BrowseComp-Plus"), ("longbench2", "LongBench v2"),
         ("mmlongdoc", "MMLongBench-Doc"), ("swe", "SWE-bench V."), ("fate", "FATE-M"))
PRIOR = ("debate", "delm", "aorchestra")
GRID = (("debate", 4, "Debate"), ("debate", 8, ""), ("debate", 16, ""),
        ("delm", 4, "DeLM"), ("delm", 8, ""), ("delm", 16, ""),
        ("aorchestra", None, "AOrchestra"),
        ("blackboard", 4, "Blackboard"), ("blackboard", 8, ""), ("blackboard", 16, ""))


def load(root: str, unit: str, task: str, suffix: str) -> dict:
    with open(os.path.join(root, f"census_{unit}", f"{task}_{unit}_{suffix}.json")) as f:
        return json.load(f)


def group_rows(d: dict) -> list:
    return [r for r in d["rows"] if not r.get("null") and not r.get("review")]


def review_rows(d: dict) -> list:
    return [r for r in d["rows"] if r.get("review") and not r.get("null") and r.get("protocol") == "blackboard"]


def reviewer_of(r: dict) -> str:
    """The reviewer variant named by the text after '+' in the cell label."""
    return r["cell"].split("+", 1)[1].strip() if "+" in r["cell"] else "reviewer"


def pick(rows: list, protocol: str, n) -> dict | None:
    if protocol == "aorchestra":
        hit = [r for r in rows if r.get("protocol") == "aorchestra"]
    else:
        hit = [r for r in rows if r.get("protocol") == protocol and r.get("n") == n]
    if len(hit) > 1:
        raise SystemExit(f"ambiguous cell {protocol} N={n}: {[h['root'] for h in hit]}")
    return hit[0] if hit else None


def fmt(r: dict | None, key: str, cls_key: str, full: int, tex: bool) -> str:
    if r is None:
        return r"\blank" if tex else "-"
    cls = r[cls_key]
    short = r["questions"] < full
    if cls == "capped":
        return "cap" + ((r"$^{\dagger}$" if tex else "*") if short else "")
    mean, se = r[key]
    if tex:
        mark = {"below": r"\downarrow", "above": r"\uparrow"}.get(cls, "")
        dag = r"\dagger" if short else ""
        sup = f"^{{{mark}{dag}}}" if (mark or dag) else ""
        return f"${mean:+.3f}{sup}$\\se{{{se:.3f}}}"
    mark = {"below": " (below)", "above": " (above)"}.get(cls, "")
    return f"{mean:+.3f} +- {se:.3f}{mark}" + ("*" if short else "")


def counts(root: str, unit: str, suffix: str, tasks) -> tuple:
    vote, truth, by_protocol, n = Counter(), Counter(), Counter(), 0
    for task, _ in tasks:
        for r in group_rows(load(root, unit, task, suffix)):
            if r.get("protocol") not in PRIOR:
                continue
            n += 1
            vote[r["class_vote"]] += 1
            truth[r["class_truth"]] += 1
            if r["class_vote"] == "below":
                by_protocol[r["protocol"]] += 1
    return n, vote, truth, by_protocol


def table(root: str, unit: str, suffix: str, tasks, tex: bool) -> str:
    data = {t: group_rows(load(root, unit, t, suffix)) for t, _ in tasks}
    full = {t: max((r["questions"] for r in rows), default=0) for t, rows in data.items()}
    reviews = {t: review_rows(load(root, unit, t, suffix)) for t, _ in tasks}
    kinds = sorted({reviewer_of(r) for rows in reviews.values() for r in rows})
    sep = " & " if tex else " | "
    end = r" \\" if tex else " |"
    start = "" if tex else "| "
    panels = []
    for key, cls_key, title in (("e_vote", "class_vote", "Accuracy minus Vote@B"),
                                ("e_truth", "class_truth", "Accuracy minus Pass@B")):
        lines = [f"\\tgroup{{8}}{{{title}}} \\\\" if tex else f"| **{title}** |" + " |" * (len(tasks) + 1)]
        for protocol, n, label in GRID:
            cells = [fmt(pick(data[t], protocol, n), key, cls_key, full[t], tex) for t, _ in tasks]
            lines.append(start + sep.join([label, str(n) if n else "auto", *cells]) + end)
        for kind in kinds:
            cells, sizes = [], set()
            for t, _ in tasks:
                hit = [r for r in reviews[t] if reviewer_of(r) == kind]
                sizes |= {r.get("n") for r in hit}
                cells.append(fmt(hit[0] if hit else None, key, cls_key, full[t], tex))
            size = "/".join(str(k) for k in sorted(s for s in sizes if s is not None))
            lines.append(start + sep.join([f"Blackboard + {kind}", size, *cells]) + end)
        panels.append("\n".join(lines))
    head = [name for _, name in tasks]
    if tex:
        return "\n".join(["\\begin{tabular}{@{}l" + "c" * (len(tasks) + 1) + "@{}}", "\\toprule",
                          "\\textbf{Method} & $N$ & " + " & ".join(f"\\textbf{{{h}}}" for h in head) + r" \\",
                          " & & " + " & ".join(f"{{\\scriptsize {full[t]} q.}}" for t, _ in tasks) + r" \\",
                          "\\midrule", panels[0], "\\midrule", panels[1], "\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(["| Method | N | " + " | ".join(head) + " |", "|" + "---|" * (len(tasks) + 2),
                      "|  |  | " + " | ".join(f"{full[t]} q." for t, _ in tasks) + " |", panels[0], panels[1], "",
                      "(above)/(below): the 95% interval lies above/below zero; *: fewer questions than the subset; "
                      "cap: the budget exceeds the pool.", ""])


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prior methods against Vote@B and Pass@B, read from the grading output "
                                             "of rci.ladder.grade (one census_<unit>/ directory per budget unit).")
    ap.add_argument("root")
    ap.add_argument("--unit", default="completion", choices=list(UNITS))
    ap.add_argument("--suffix", default="strict_census_taskrng")
    ap.add_argument("--tasks", default=None, help="comma list of spec task names (default: the six public tasks)")
    ap.add_argument("--out", default=None, help="output file; .tex writes LaTeX, anything else Markdown")
    a = ap.parse_args(argv)
    names = dict(TASKS)
    tasks = tuple((t, names.get(t, t)) for t in a.tasks.split(",")) if a.tasks else TASKS
    for unit in UNITS:
        if not os.path.isdir(os.path.join(a.root, f"census_{unit}")):
            continue
        n, vote, truth, by_protocol = counts(a.root, unit, a.suffix, tasks)
        v = "/".join(str(vote[c]) for c in ("above", "below", "undetermined", "capped"))
        t = "/".join(str(truth[c]) for c in ("above", "below", "undetermined", "capped"))
        m = ", ".join(f"{k} {by_protocol[k]}" for k in PRIOR)
        print(f"{unit:<10} n={n}  vs Vote@B {v}  vs Pass@B {t}  below Vote@B by protocol: {m}")
    text = table(a.root, a.unit, a.suffix, tasks, tex=bool(a.out and a.out.endswith(".tex")))
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
