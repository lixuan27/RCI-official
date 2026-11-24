"""SplitBench train / held-out split by hashed ids, and sealing of the writer's answer key."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rci import paths
from rci.tasks.hiddenbench.task import norm

KEY_FIELDS = ("writer_answer", "decoy", "explanation")


def qid(row: dict) -> str:
    return f"hb_{row['id']:02d}"


def split_ids(ids, heldout_fraction: float = 196 / 697, salt: str = "splitbench") -> tuple[list[str], list[str]]:
    """(train, held-out): ids ranked by sha256(salt|id); the first round(fraction * n) are held out."""
    ids = sorted(set(map(str, ids)))
    ranked = sorted(ids, key=lambda q: hashlib.sha256(f"{salt}|{q}".encode()).hexdigest())
    k = int(round(heldout_fraction * len(ids)))
    held = set(ranked[:k])
    return [q for q in ids if q not in held], [q for q in ids if q in held]


def write_bench(directory: Path, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "benchmark.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")


def seal(round_dir) -> dict:
    """Move the writer's key into writer_key.json and score the round by the full-information answer from then on."""
    round_dir = Path(round_dir)
    p = round_dir / "benchmark.json"
    rows = json.loads(p.read_text())
    if all("writer_answer" not in r for r in rows):
        return {"sealed": 0, "already_sealed": True}
    key = {qid(r): {k: r.get(k) for k in KEY_FIELDS} for r in rows}
    (round_dir / "writer_key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n")
    sealed = [{**{k: v for k, v in r.items() if k not in KEY_FIELDS + ("matches_writer",)},
               "correct_answer": r["full_answer"]} for r in rows]
    p.write_text(json.dumps(sealed, indent=1, ensure_ascii=False) + "\n")
    differs = sum(norm(v["writer_answer"]) != norm(r["full_answer"]) for r, v in zip(rows, key.values()))
    return {"sealed": len(sealed), "key_differs_from_full_answer": differs}


def cmd_split(a) -> None:
    base = Path(a.root or paths.data_dir("splitbench"))
    rows = json.loads((Path(a.admitted or base / "admission") / "benchmark.json").read_text())
    by_id = {qid(r): r for r in rows}
    if a.ids_file:
        spec = json.loads(Path(a.ids_file).read_text())
        train, held = list(spec["train"]), list(spec["heldout"])
        missing = sorted(set(train + held) - set(by_id))
        if missing:
            raise SystemExit(f"[split] {len(missing)} ids of --ids-file are not admitted, e.g. {missing[:3]}")
    else:
        train, held = split_ids(by_id, a.heldout_fraction, a.salt)
    write_bench(base / "train", [by_id[q] for q in train])
    write_bench(base / "heldout", [by_id[q] for q in held])
    out = {"train": len(train), "heldout": len(held)}
    if not a.no_seal:
        out["seal"] = {d: seal(base / d) for d in ("train", "heldout")}
    print(json.dumps(out))


def cmd_seal(a) -> None:
    print(json.dumps(seal(a.round)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.splitbench.split",
                                 description="Split admitted SplitBench tasks into train and held-out sets and seal them.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split", help="write <root>/train and <root>/heldout from <root>/admission, then seal both")
    s.add_argument("--root", default=None, help="default: $RCI_DATA/splitbench (splitbench_control for the control set)")
    s.add_argument("--admitted", default=None, help="directory of the admitted benchmark.json (default: <root>/admission)")
    s.add_argument("--heldout-fraction", type=float, default=196 / 697)
    s.add_argument("--salt", default="splitbench")
    s.add_argument("--ids-file", default=None, help='JSON {"train": [ids], "heldout": [ids]} reproducing a split exactly')
    s.add_argument("--no-seal", action="store_true")
    s = sub.add_parser("seal", help="seal one round directory")
    s.add_argument("--round", required=True)
    a = ap.parse_args(argv)
    {"split": cmd_split, "seal": cmd_seal}[a.cmd](a)


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
