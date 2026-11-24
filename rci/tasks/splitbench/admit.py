"""SplitBench admission from the members' own behaviour under three profiles, never from the writer's key."""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from rci import paths


def answers(root: str, n: int = 4) -> dict[str, list[str]]:
    """Question -> every member answer of the independent runs under `root`."""
    out: dict[str, list[str]] = defaultdict(list)
    for p in sorted(glob.glob(f"{root}/independent/N{n}/s*/*.json.gz")):
        rec = json.load(gzip.open(p, "rt"))
        out[rec["example_id"]] += [x for x in rec["score"]["preds"] if x]
    return out


def admit(rows: list[dict], full: dict, shared: dict, hidden: dict, agree: float, shared_max: float, hidden_max: float,
          control: bool = False):
    norm = lambda x: " ".join(str(x or "").lower().split()).strip(" .,:;!?\"'`")
    kept, why = [], Counter()
    for r in rows:
        q = f"hb_{r['id']:02d}"
        f, s, h = full.get(q, []), shared.get(q, []), hidden.get(q, [])
        if not f or not s or (not h and not control):
            why["not run"] += 1
            continue
        a, n = Counter(f).most_common(1)[0]
        if n / len(f) < agree:
            why["full information disagrees"] += 1
            continue
        if control:
            if s.count(a) / len(s) < 1 - shared_max:
                why["shared facts do not point to the answer"] += 1
                continue
        elif s.count(a) / len(s) > shared_max or Counter(s).most_common(1)[0][0] == a:
            why["shared facts do not mislead"] += 1
            continue
        if not control and h.count(a) / len(h) > hidden_max:
            why["solvable without pooling"] += 1
            continue
        spelled = {norm(o): o for o in r["possible_answers"]}
        kept.append({**r, "full_answer": spelled.get(a, a), "full_agree": round(n / len(f), 3),
                     "matches_writer": norm(a) == norm(r["writer_answer"])})
    return kept, why


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.splitbench.admit",
                                 description="Admit SplitBench candidates from independent runs under three profiles.")
    ap.add_argument("--round", default=None, help="candidate directory (default: $RCI_DATA/splitbench/raw)")
    ap.add_argument("--results", required=True, help="directory holding the full/, shared/ and hidden/ runs")
    ap.add_argument("--out", default=None, help="default: <round>/../admission")
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--agree", type=float, default=0.75)
    ap.add_argument("--shared-max", type=float, default=0.25)
    ap.add_argument("--hidden-max", type=float, default=0.40)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args(argv)
    rnd = Path(a.round or paths.data_dir("splitbench_control" if a.control else "splitbench", "raw"))
    out = Path(a.out) if a.out else rnd.parent / "admission"
    rows = json.loads((rnd / "benchmark.json").read_text())
    kept, why = admit(rows, *(answers(f"{a.results}/{p}", a.n_agents) for p in ("full", "shared", "hidden")),
                      a.agree, a.shared_max, a.hidden_max, a.control)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False) + "\n")
    rep = {"candidates": len(rows), "admitted": len(kept), "dropped": dict(why),
           "full_answer_matches_writer": round(sum(k["matches_writer"] for k in kept) / max(1, len(kept)), 3)}
    (out / "ADMIT.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep))


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
