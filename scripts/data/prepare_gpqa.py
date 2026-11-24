"""Lay out GPQA-Diamond (198 questions) for rci.tasks.gpqa from the authors' archive at a pinned commit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

from rci import paths

REVISION = "d46dc8d5e01b40bcde0bed6bee68a5de953a58f8"
URL = f"https://github.com/idavidrein/gpqa/raw/{REVISION}/dataset.zip"
ZIP_SHA256 = "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc"
PASSWORD = b"deserted-untie-orchid"
LETTERS = "ABCD"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download GPQA-Diamond at a pinned commit and write samples.jsonl.")
    ap.add_argument("--zip", default=None, help="a local copy of dataset.zip (default: download it)")
    ap.add_argument("--out", default=None, help="default: $RCI_DATA/gpqa")
    a = ap.parse_args()
    a.out = a.out or str(paths.data_dir("gpqa"))
    if a.zip:
        raw = Path(a.zip).read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=120) as r:
            raw = r.read()
    if hashlib.sha256(raw).hexdigest() != ZIP_SHA256:
        raise SystemExit("dataset.zip sha256 mismatch: not the pinned archive")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read("dataset/gpqa_diamond.csv", pwd=PASSWORD).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 198, len(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "samples.jsonl", "w") as f:
        for i, r in enumerate(rows):
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            order = list(range(4))
            random.Random(f"gpqa:{r['Record ID']}").shuffle(order)
            f.write(json.dumps({"idx": i, "id": r["Record ID"], "question": r["Question"].strip(),
                                "choices": [opts[j].strip() for j in order], "answer": LETTERS[order.index(0)],
                                "domain": r.get("High-level domain"), "subdomain": r.get("Subdomain")}) + "\n")
    (out / "VERSION").write_text(f"idavidrein/gpqa@{REVISION} dataset.zip sha256 {ZIP_SHA256} gpqa_diamond.csv\n")
    print(f"wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    main()
