"""Download LongBench v2 at a pinned revision and lay it out for rci.tasks.longbench2 (samples.jsonl and ctx/)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

from rci import paths

REPO, REVISION = "THUDM/LongBench-v2", "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
META = ("_id", "domain", "sub_domain", "difficulty", "length", "question", "choice_A", "choice_B", "choice_C",
        "choice_D", "answer")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download LongBench v2 at a pinned revision and write samples.jsonl and ctx/.")
    ap.add_argument("--out", default=None, help="default: $RCI_DATA/longbench2")
    a = ap.parse_args()
    out = Path(a.out or paths.data_dir("longbench2"))
    (out / "ctx").mkdir(parents=True, exist_ok=True)
    path = out / "data.json"
    url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}/datasets/{REPO}/resolve/{REVISION}/data.json"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as r, open(f"{path}.part", "wb") as f:
            for block in iter(lambda: r.read(1 << 22), b""):
                f.write(block)
        os.replace(f"{path}.part", path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != SHA256:
        raise SystemExit(f"sha256 mismatch: {h.hexdigest()} != {SHA256}: not the pinned file")
    rows = json.loads(path.read_text())
    assert len(rows) == 503, len(rows)
    with open(out / "samples.jsonl", "w") as f:
        for i, r in enumerate(rows):
            (out / "ctx" / f"{r['_id']}.txt").write_text(r["context"])
            f.write(json.dumps({"idx": i, **{k: r[k] for k in META}, "n_chars": len(r["context"])}) + "\n")
    (out / "VERSION").write_text(f"{REPO}@{REVISION} data.json sha256 {SHA256}\n")
    print(f"wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    main()
