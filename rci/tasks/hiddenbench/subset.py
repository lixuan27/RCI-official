"""The HiddenBench task subset with exactly n private facts whose facts are told apart by their keys."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

from rci import paths
from rci.tasks.hiddenbench.facts import holders, holds, keys_of, key_collisions, message_text, multi_fact_errors, words
from rci.tasks.hiddenbench.task import FORWARD_RATIONALE, FORWARD_TITLE

EXCLUDE = {"hb_07": "facts are short table rows: forward_facts refuses them"}


def forward_artifact(facts: list[str]) -> str:
    """The artifact forward_facts stores for these facts."""
    return "\n".join(f"- {x}" for x in facts)


def forward_text(facts: list[str]) -> str:
    """A forward as the structure reader and holds_every_fact match it: title | artifact | rationale."""
    return message_text({"sub_task": FORWARD_TITLE, "artifact": forward_artifact(facts), "rationale": FORWARD_RATIONALE})


def forward_errors(row: dict, n: int, sizes=(1, 2, 3)) -> tuple[list[str], list[str]]:
    """(errors, warnings) for forwards of 1-3 private facts, alone or with the shared facts, read back by key."""
    keys, hidden, shared = keys_of(row), list(row["hidden_information"]), list(row["shared_information"])
    errors, warnings = [], []
    cases = [((), True)] if shared else []
    cases += [(c, w) for k in sizes for c in combinations(range(len(hidden)), k) for w in ((False, True) if shared else (False,))]
    for S, with_shared in cases:
        facts = (shared if with_shared else []) + [hidden[i] for i in S]
        what = f"{'shared facts + ' if with_shared else ''}private {list(S)}" if S else "shared facts alone"
        for form, text in (("full message", forward_text(facts)), ("artifact", forward_artifact(facts))):
            got = sorted(h for h, k in enumerate(keys) if holds(words(text), k))
            if got != list(S):
                errors.append(f"forward of {what} holds {got} by its {form}")
        _, flag = holders(row, forward_artifact(facts), n)
        if flag != with_shared:
            warnings.append(f"forward of {what}: header says shared facts {flag}")
    return errors, warnings


def check(rows: list[dict], n: int, exclude: dict[str, str] = EXCLUDE) -> dict:
    kept, dropped, warnings = [], {}, {}
    by_check = {"excluded": 0, "a_key_collision": 0, "b_multi_fact": 0, "b_forward_form_only": 0}
    with_n = [r for r in rows if len(r["hidden_information"]) == n]
    for r in sorted(with_n, key=lambda r: r["id"]):
        q = r["id"]
        if q in exclude:
            dropped[q] = [f"excluded: {exclude[q]}"]
            by_check["excluded"] += 1
            continue
        why = []
        keys = keys_of(r)
        coll = key_collisions(r)
        for i, j in coll:
            why.append(f"key collision: fact {j} holds fact {i}'s key {sorted(keys[i])}")
        multi = multi_fact_errors(r)
        for s in multi:
            w = words("\n".join(r["hidden_information"][i] for i in s))
            got = sorted(h for h, k in enumerate(keys) if holds(w, k))
            why.append(f"multi-fact message {list(s)} holds {got}")
        errs, warns = forward_errors(r, n)
        why += errs
        if warns:
            warnings[q] = warns
        if why:
            dropped[q] = why
            by_check["a_key_collision"] += bool(coll)
            by_check["b_multi_fact"] += bool(multi)
            by_check["b_forward_form_only"] += not coll and not multi
            continue
        kept.append(q)
    return {"questions": kept, "dropped": dropped, "warnings": warnings,
            "counts": {"rows": len(rows), f"with_{n}_private": len(with_n), "kept": len(kept), "dropped": len(dropped),
                       "dropped_by_check": by_check}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.hiddenbench.subset",
                                 description="Select the HiddenBench tasks with exactly n private facts that pass the key "
                                             "checks.")
    ap.add_argument("--data", default=None, help="benchmark.json (default: $RCI_DATA/hiddenbench/benchmark.json)")
    ap.add_argument("--n", type=int, default=4, help="private facts per question (= members)")
    ap.add_argument("--out", default="configs/hiddenbench/tasks56.json")
    a = ap.parse_args(argv)
    a.data = a.data or str(paths.data_dir("hiddenbench", "benchmark.json"))
    if not Path(a.data).exists():
        raise FileNotFoundError(f"{a.data} missing: run scripts/data/prepare_hiddenbench.py")
    raw = Path(a.data).read_bytes()
    rows = [{**r, "id": f"hb_{r['id']:02d}"} for r in json.loads(raw)]
    res = check(rows, a.n)
    for q, why in res["dropped"].items():
        print(f"[subset] drop {q}: " + "; ".join(why), flush=True)
    for q, why in res["warnings"].items():
        print(f"[subset] warning {q} (kept): " + "; ".join(why), flush=True)
    print(f"[subset] {res['counts']}", flush=True)
    out = {"n_hidden": a.n, "data_sha256": hashlib.sha256(raw).hexdigest(), **res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"[subset] wrote {a.out}: {len(res['questions'])} questions", flush=True)


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
