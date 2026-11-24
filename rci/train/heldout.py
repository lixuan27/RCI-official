"""Held-out contradiction-audit metrics of trained groups, paired against the untrained model."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from rci.ladder.records import load
from rci.paths import data_dir

METRICS = ("recall", "f1", "precision", "lenient_recall", "claims")


def instance(example_id: str, root: Path, cache: dict) -> dict | None:
    """The audit instance of an example id '<project>_x<level>', read from <root>/x<level>/<project>.json."""
    if example_id not in cache:
        proj, lv = str(example_id).rsplit("_x", 1)
        p = root / f"x{lv}" / f"{proj}.json"
        cache[example_id] = json.loads(p.read_text()) if p.exists() else None
    return cache[example_id]


def pairs_of(rec: dict) -> set:
    """Distinct snippet pairs reported by the group (the union of its members' claims)."""
    out = set()
    for f in rec.get("finals") or []:
        if f.get("failing"):
            continue
        for p in ((f.get("final") or {}).get("pairs") or []):
            try:
                a, b = int(p.get("a")), int(p.get("b"))
            except (TypeError, ValueError, AttributeError):
                continue
            if a != b:
                out.add(frozenset((a, b)))
    return out


def metrics(rec: dict, ex: dict) -> dict:
    """Exact-pair recall, F1 and precision, the lenient benchmark recall, and claims per run."""
    con = ex["contradictions"]
    true = {frozenset((c["edited"], c["partner"])) for c in con}
    rep = pairs_of(rec)
    hit = len(rep & true)
    covered = {x for q in rep for x in q}
    return {"recall": hit / max(1, len(true)), "f1": 2 * hit / (len(rep) + len(true)) if (rep or true) else 0.0,
            "precision": hit / len(rep) if rep else 0.0,
            "lenient_recall": sum(1 for c in con if c["edited"] in covered) / max(1, len(con)), "claims": len(rep)}


def cell(root: str, protocol: str, n: int, eval_root: Path, cache: dict) -> dict:
    """(example id, seed) -> metrics for every record of one evaluation cell."""
    out = {}
    for ex_id, recs in load(root, protocol, n).items():
        for rec in recs:
            ex = instance(rec["example_id"], eval_root, cache)
            if ex is None:
                continue
            out[(str(rec["example_id"]), (rec.get("config") or {}).get("seed", 0))] = metrics(rec, ex)
    return out


def paired_bootstrap(diffs: dict, boot: int, seed: int) -> tuple:
    """Mean difference and 95% interval, resampling examples with all their seeds."""
    rng = random.Random(seed)
    ex = list(diffs)
    flat = [d for ds in diffs.values() for d in ds]
    means = []
    for _ in range(boot):
        sample = [d for _ in ex for d in diffs[ex[rng.randrange(len(ex))]]]
        means.append(sum(sample) / len(sample))
    means.sort()
    return sum(flat) / len(flat), means[int(0.025 * boot)], means[int(0.975 * boot) - 1]


def compare(ref: dict, got: dict, boot: int, seed: int) -> dict:
    keys = sorted(set(ref) & set(got))
    row = {"pairs": len(keys), "examples": len({k[0] for k in keys})}
    if not keys:
        return row
    for m in METRICS:
        scale = 1.0 if m == "claims" else 100.0
        by_ex = defaultdict(list)
        for k in keys:
            by_ex[k[0]].append(scale * (got[k][m] - ref[k][m]))
        d, lo, hi = paired_bootstrap(by_ex, boot, seed)
        row[m] = scale * sum(got[k][m] for k in keys) / len(keys)
        row[f"{m}_ref"] = scale * sum(ref[k][m] for k in keys) / len(keys)
        row[f"{m}_diff"] = {"mean": d, "lo": lo, "hi": hi, "excludes_zero": lo > 0 or hi < 0}
    return row


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Held-out recall, F1 and precision of audit groups, each run paired "
                                             "against a reference cell (the untrained model) with a bootstrap over "
                                             "corpora.")
    ap.add_argument("--ref", required=True, help="NAME=ROOT of the reference cell")
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=ROOT of each trained cell")
    ap.add_argument("--protocol", default="blackboard")
    ap.add_argument("--n", default="4,8", help="comma list of group sizes")
    ap.add_argument("--eval-data", default=None, help="audit evaluation instances (default: $RCI_DATA/audit/eval)")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    eval_root = Path(a.eval_data) if a.eval_data else data_dir("audit", "eval")
    cache: dict = {}
    ref_name, ref_root = a.ref.split("=", 1)
    res = {"reference": ref_name, "protocol": a.protocol, "boot": a.boot, "rows": []}
    lines = ["| run | N | pairs | recall | F1 | precision | claims | d recall [95% CI] | d F1 [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n in (int(x) for x in a.n.split(",")):
        ref = cell(ref_root, a.protocol, n, eval_root, cache)
        for spec in a.runs:
            name, root = spec.split("=", 1)
            row = {"run": name, "n": n, **compare(ref, cell(root, a.protocol, n, eval_root, cache), a.boot, a.seed)}
            res["rows"].append(row)
            if not row["pairs"]:
                lines.append(f"| {name} | {n} | 0 | - | - | - | - | - | - |")
                continue

            def ci(m):
                d = row[f"{m}_diff"]
                return f"{d['mean']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]" + (" *" if d["excludes_zero"] else "")

            lines.append(f"| {name} | {n} | {row['pairs']} | {row['recall']:.1f} | {row['f1']:.1f} | "
                         f"{row['precision']:.1f} | {row['claims']:.1f} | {ci('recall')} | {ci('f1')} |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout.json").write_text(json.dumps(res, indent=1))
    (out / "heldout.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
