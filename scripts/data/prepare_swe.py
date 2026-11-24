"""Download SWE-bench Verified at a pinned revision and pull the official images of its first instances."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from rci import paths

REPO, REVISION = "SWE-bench/SWE-bench_Verified", "78f471bf655a3137b2e8a75af1501690ec009ec3"
FILE, SHA256 = "data/test-00000-of-00001.parquet", "030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25"
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download SWE-bench Verified and its evaluation images (sha256-verified).")
    ap.add_argument("--first", type=int, default=100, help="instances whose images are pulled")
    ap.add_argument("--images", type=int, default=4, help="images pulled concurrently")
    ap.add_argument("--jobs", type=int, default=16, help="concurrent HTTP connections")
    ap.add_argument("--skip-images", action="store_true")
    a = ap.parse_args(argv)
    root = paths.data_dir("swe")
    parquet = root / "hf" / "verified.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    if not parquet.exists() or sha256(parquet) != SHA256:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        with urllib.request.urlopen(f"{endpoint}/datasets/{REPO}/resolve/{REVISION}/{FILE}", timeout=300) as r:
            parquet.write_bytes(r.read())
    if sha256(parquet) != SHA256:
        raise SystemExit("verified.parquet: sha256 mismatch")
    if a.skip_images:
        return 0
    import importlib.util

    import pyarrow.parquet as pq
    spec = importlib.util.spec_from_file_location("oci_pull", HERE / "oci_pull.py")
    oci_pull = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oci_pull)

    rows = pq.read_table(str(parquet)).to_pylist()[: a.first]
    store = root / "oci"
    store.mkdir(parents=True, exist_ok=True)
    digests_path = store / "official_digests.json"
    digests = json.loads(digests_path.read_text()) if digests_path.exists() else {}
    for r in rows:
        if r["instance_id"] not in digests:
            digests[r["instance_id"]] = oci_pull.official_digest(r["image"])
    digests_path.write_text(json.dumps(digests, indent=1))
    refs = store / f"refs_first{a.first}.tsv"
    with open(refs, "w") as f:
        for r in rows:
            name = r["image"].rsplit(":", 1)[0]
            f.write(f"{r['instance_id']}\t{name}@{digests[r['instance_id']]['platform_digest']}\n")
    rc = subprocess.call([sys.executable, str(HERE / "oci_pull.py"), "--store", str(store), "--list", str(refs),
                          "--images", str(a.images), "--jobs", str(a.jobs)])
    report = {}
    for r in rows:
        key = r["instance_id"]
        if not (store / "images" / f"{key}.json").exists():
            report[key] = {"problems": ["not pulled"], "passed": False}
            continue
        report[key] = oci_pull.verify_image(str(store), key, digests[key])
    (store / f"verify_first{a.first}.json").write_text(json.dumps(report, indent=1))
    bad = [k for k, v in report.items() if v.get("problems")]
    print(f"verified {len(report) - len(bad)}/{len(report)} images")
    return 1 if (rc or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
