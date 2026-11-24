"""Training dynamics from rollout files: claims per run and their precision over steps, and credit spread."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

COLUMNS = ("score", "claims", "exact", "precision", "pooled_precision", "member", "commit", "participation",
           "credit_gini", "messages", "ktok")


def gini(xs: list) -> float | None:
    xs = sorted(x - min(xs) for x in xs) if xs else []
    n, s = len(xs), sum(xs)
    if n < 2 or s <= 0:
        return 0.0 if n >= 2 else None
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs)) / (n * s)


def rows(run_dir: str):
    """(step, per-run summary) for every training rollout of a run."""
    for f in sorted(glob.glob(os.path.join(run_dir, "rollouts", "step_*", "*.jsonl"))):
        try:
            with open(f) as fh:
                head = json.loads(fh.readline())
        except (OSError, ValueError):
            continue
        step = int(os.path.basename(os.path.dirname(f)).split("_")[1])
        for s in head.get("summaries") or []:
            cred = [float(v) for v in (s.get("credits") or {}).values()]
            comp = s.get("compute") or {}
            yield step, {"score": s.get("score"), "claims": s.get("distinct_pairs"), "exact": s.get("exact_pairs"),
                         "precision": s.get("precision"), "member": s.get("per_agent_mean"),
                         "commit": s.get("vote_stake"),
                         "participation": (sum(c > 1e-9 for c in cred) / len(cred)) if cred else None,
                         "credit_gini": gini(cred), "messages": s.get("n_messages"),
                         "ktok": (comp.get("total_tokens") or 0) / 1000 if comp else None}


def summarise(recs: list) -> dict:
    """Means over runs; pooled precision is exact pairs over claims summed across runs."""
    out = {}
    for c in COLUMNS:
        if c == "pooled_precision":
            claims = sum(r["claims"] or 0 for r in recs if r["claims"] is not None and r["exact"] is not None)
            exact = sum(r["exact"] or 0 for r in recs if r["claims"] is not None and r["exact"] is not None)
            out[c] = exact / claims if claims else None
            continue
        vals = [r[c] for r in recs if r.get(c) is not None]
        out[c] = sum(vals) / len(vals) if vals else None
    out["runs"] = len(recs)
    return out


def span(text: str) -> tuple:
    lo, _, hi = text.partition("-")
    return int(lo), int(hi or lo)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Claims per run and precision over training steps (windows, and an early "
                                             "and a late range of steps) for one or more training runs.")
    ap.add_argument("--runs", nargs="+", required=True, help="training run directories (rci.train.loop output)")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--early", default="1-5", help="step range, e.g. 1-5")
    ap.add_argument("--late", default=None, help="step range (default: the last --window steps)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {}
    lines = ["| run | range | runs | claims | precision | pooled precision | score |", "|---|---|---|---|---|---|---|"]
    for run in a.runs:
        by_step = defaultdict(list)
        for step, r in rows(run):
            by_step[step].append(r)
        if not by_step:
            continue
        last = max(by_step)
        windows = {}
        for w0 in range(1, last + 1, a.window):
            got = [r for s in range(w0, min(last, w0 + a.window - 1) + 1) for r in by_step.get(s, [])]
            if got:
                windows[f"{w0}-{min(last, w0 + a.window - 1)}"] = summarise(got)
        late = a.late or f"{max(1, last - a.window + 1)}-{last}"
        ranges = {}
        for name, text in (("early", a.early), ("late", late)):
            lo, hi = span(text)
            ranges[name] = {"steps": text, **summarise([r for s in range(lo, hi + 1) for r in by_step.get(s, [])])}

            def f(x):
                return "-" if x is None else f"{x:.3f}"

            g = ranges[name]
            lines.append(f"| {Path(run).name} | {name} {text} | {g['runs']} | {f(g['claims'])} | {f(g['precision'])} | "
                         f"{f(g['pooled_precision'])} | {f(g['score'])} |")
        res[Path(run).name] = {"windows": windows, "ranges": ranges}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "curves.json").write_text(json.dumps(res, indent=1))
    (out / "curves.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
