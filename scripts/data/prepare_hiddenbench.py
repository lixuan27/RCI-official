"""Download the 65 HiddenBench tasks at a pinned commit and verify them against the hash the experiments used."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from rci import paths

REPO = "Yassellee/HiddenBench_ICML"
COMMIT = "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1"
FILE = "data/benchmark.json"
URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{FILE}"
SHA256 = "2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download HiddenBench (MIT) to $RCI_DATA/hiddenbench/benchmark.json.")
    ap.add_argument("--out", default=None, help="default: $RCI_DATA/hiddenbench")
    a = ap.parse_args(argv)
    out = Path(a.out or paths.data_dir("hiddenbench"))
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as r:
        raw = r.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != SHA256:
        print(f"sha256 {got} differs from the pinned {SHA256}", file=sys.stderr)
        return 1
    rows = json.loads(raw)
    if len(rows) != 65:
        print(f"expected 65 tasks, got {len(rows)}", file=sys.stderr)
        return 1
    (out / "benchmark.json").write_bytes(raw)
    (out / "VERSION").write_text(f"{REPO}@{COMMIT} {FILE} sha256 {SHA256}\n")
    print(f"wrote {len(rows)} tasks to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
