"""The HiddenBench chain's task sets, frozen once from hidden-profile tasks the base model wrote and admitted.

Input: admitted task files of rci.tasks.splitbench.admit (hidden type and control type), each row with `full_answer`,
the base model's own full-information agreed answer, and `full_agree`, its share among full-information members. A
task is kept when that share reaches the reference agreement and the answer is exactly one option; its
`correct_answer` becomes that reference, so a record's own score is agreement with the reference and never with the
writer's key. Near-duplicates (word-set Jaccard above 0.5, in a hash order) are dropped.

    select   the first 200 hidden-type and 100 control-type tasks in a hash order (selection and confirmation)
    pool     the rest, dealt in blocks of 20 positions (11 hidden, 9 control); round r harvests positions
             [400 (r - 1), 400 r) plus the questions still on its frontier

    python -m rci.chains.hiddenbench.tasks --hidden DIR --control DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rci.chains.util import sha, write_json

SALT = "hiddenbench/"
N_SELECT = {"hidden": 200, "control": 100}
BLOCK = {"hidden": 11, "control": 9}


def words(text: str) -> set:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def text_of(r: dict) -> str:
    return " ".join([str(r.get("description", ""))] + list(r.get("shared_information", []))
                    + list(r.get("hidden_information", [])))


def dedup(rows: list, thr: float = 0.5) -> tuple:
    kept, dropped, seen = [], [], []
    for r in sorted(rows, key=lambda r: _h("dedup/", r)):
        w = words(text_of(r))
        if any(len(w & s) / max(1, len(w | s)) > thr for s in seen):
            dropped.append(r)
            continue
        seen.append(w)
        kept.append(r)
    return kept, dropped


def _h(tag: str, r: dict) -> str:
    return sha(f"{SALT}{tag}{r['id']}")


def deal(hidden: list, control: list) -> list:
    out, h, c = [], list(hidden), list(control)
    while len(h) >= BLOCK["hidden"] and len(c) >= BLOCK["control"]:
        blk = h[:BLOCK["hidden"]] + c[:BLOCK["control"]]
        h, c = h[BLOCK["hidden"]:], c[BLOCK["control"]:]
        out += sorted(blk, key=lambda r: _h("block/", r))
    return out


def freeze(hidden_rows: list, control_rows: list, agree: float) -> dict:
    rows, why = [], {"agreement below the reference line": 0, "answer is not exactly one option": 0}
    for r, ctl in [(x, False) for x in hidden_rows] + [(x, True) for x in control_rows]:
        if float(r.get("full_agree", 0)) < agree - 1e-9:
            why["agreement below the reference line"] += 1
        elif list(r["possible_answers"]).count(r["full_answer"]) != 1:
            why["answer is not exactly one option"] += 1
        else:
            rows.append({**{k: v for k, v in r.items() if k not in ("writer_answer", "full_answer", "matches_writer")},
                         "control": ctl, "correct_answer": r["full_answer"]})
    ids = [int(r["id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("the hidden and control task files share ids")
    kept, dup = dedup(rows)
    hid = sorted((r for r in kept if not r["control"]), key=lambda r: _h("select/", r))
    ctl = sorted((r for r in kept if r["control"]), key=lambda r: _h("select/", r))
    select = hid[:N_SELECT["hidden"]] + ctl[:N_SELECT["control"]]
    pool_h = sorted(hid[N_SELECT["hidden"]:], key=lambda r: _h("pool/", r))
    pool_c = sorted(ctl[N_SELECT["control"]:], key=lambda r: _h("pool/", r))
    pool = deal(pool_h, pool_c)
    return {"select": sorted(select, key=lambda r: int(r["id"])), "pool": pool,
            "report": {"kept": len(kept), "near_duplicates": len(dup), "dropped": why, "select": len(select),
                       "pool": len(pool), "reference_agreement": agree}}


def example_id(r: dict) -> str:
    return f"hb_{int(r['id']):02d}"


def write(out: Path, frozen: dict) -> dict:
    for name in ("select", "pool"):
        rows = sorted(frozen[name], key=lambda r: int(r["id"]))
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "benchmark.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    write_json(out / "pool_order.json", [example_id(r) for r in frozen["pool"]])
    write_json(out / "control.json", sorted(example_id(r) for r in frozen["select"] + frozen["pool"] if r["control"]))
    write_json(out / "FROZEN.json", frozen["report"])
    return frozen["report"]


def slice_ids(out: Path, r: int, size: int) -> list:
    pool = json.loads((out / "pool_order.json").read_text())
    return pool[size * (r - 1): size * r]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rci.chains.hiddenbench.tasks", description=__doc__.splitlines()[0])
    ap.add_argument("--hidden", required=True, help="admitted hidden-type tasks (a directory with benchmark.json)")
    ap.add_argument("--control", required=True, help="admitted control-type tasks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agree", type=float, default=0.75)
    a = ap.parse_args(argv)
    load = lambda d: json.loads((Path(d) / "benchmark.json").read_text())  # noqa: E731
    print(json.dumps(write(Path(a.out), freeze(load(a.hidden), load(a.control), a.agree))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
