"""Download MMLongBench-Doc at pinned revisions and pre-render it: page text layers, page images and a manifest."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rci import paths

RENDER = {"dpi_cap": 144, "max_pixels": 1600 * 32 * 32, "max_side": 2048, "jpeg_quality": 90, "token_px": 32,
          "text_mode": "text,sort"}
N_SAMPLES = 1082
GITHUB_COMMIT = "d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384"
GITHUB = f"https://raw.githubusercontent.com/mayubo2333/MMLongBench-Doc/{GITHUB_COMMIT}"
HF_REPO, HF_REVISION = "yubo2333/MMLongBench-Doc", "2ff6aa9237fc777b6627dc57a486e9225ac5fb86"
PROMPT_FILE = "prompt_for_answer_extraction.md"
SAMPLES_PATH = "data/samples.json"
SAMPLES_SHA256 = "4ffadad60b18cd7ede864155f3e8a366fe4deec2189b3d0a25691b7df4b839e8"


def render_zoom(w_pt: float, h_pt: float, r: dict = RENDER) -> float:
    z = r["dpi_cap"] / 72.0
    if w_pt > 0 and h_pt > 0:
        z = min(z, math.sqrt(r["max_pixels"] / (w_pt * h_pt)), r["max_side"] / max(w_pt, h_pt))
    return z


def vision_tokens(w: int, h: int, px: int = RENDER["token_px"]) -> int:
    return max(1, round(w / px)) * max(1, round(h / px))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    os.replace(tmp, dest)


def fetch(raw: Path, out: Path) -> None:
    if not (raw / "samples.json").exists():
        download(GITHUB + "/" + SAMPLES_PATH, raw / "samples.json")
    if not (out / PROMPT_FILE).exists():
        download(f"{GITHUB}/eval/{PROMPT_FILE}", out / PROMPT_FILE)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    with urllib.request.urlopen(f"{endpoint}/api/datasets/{HF_REPO}/tree/{HF_REVISION}/documents", timeout=120) as r:
        tree = json.load(r)
    bad = []
    for item in tree:
        if item.get("type") != "file" or not item["path"].endswith(".pdf"):
            continue
        dest = raw / item["path"]
        oid = (item.get("lfs") or {}).get("oid")
        for _ in range(3):
            if dest.exists() and (oid is None or sha256(dest) == oid):
                break
            download(f"{endpoint}/datasets/{HF_REPO}/resolve/{HF_REVISION}/{item['path']}", dest)
        if not dest.exists() or (oid is not None and sha256(dest) != oid):
            bad.append(item["path"])
    (raw / "INTEGRITY.json").write_text(json.dumps({"source": f"{HF_REPO}@{HF_REVISION}",
                                                    "verified_against": "LFS sha256", "unverified": bad}, indent=1))
    if bad:
        sys.exit(f"[prepare] {len(bad)} documents failed verification, e.g. {bad[:3]}")


def find_samples(raw: Path) -> Path:
    for c in (raw / "samples.json", raw / "data" / "samples.json"):
        if c.exists():
            return c
    sys.exit(f"[prepare] no samples.json under {raw}")


def find_documents(raw: Path) -> Path:
    for c in (raw / "documents", raw / "data" / "documents"):
        if c.is_dir():
            return c
    sys.exit(f"[prepare] no documents/ under {raw}")


def load_samples(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    out = []
    for i, s in enumerate(rows):
        out.append({"id": f"mmld{i:04d}", "idx": i, "doc_id": s["doc_id"], "doc_type": s["doc_type"],
                    "question": s["question"], "answer": s["answer"], "answer_format": s["answer_format"],
                    "evidence_pages": list(ast.literal_eval(s["evidence_pages"])),
                    "evidence_sources": list(ast.literal_eval(s["evidence_sources"]))})
    return out


def doc_stem(doc_id: str) -> str:
    return doc_id[:-4] if doc_id.lower().endswith(".pdf") else doc_id


def prepare_doc(pdf: str, out_dir: str, render: dict) -> dict:
    import pymupdf
    from PIL import Image

    pdf_p, out = Path(pdf), Path(out_dir)
    digest = sha256(pdf_p)
    meta_p = out / "meta.json"
    if meta_p.exists():
        try:
            old = json.loads(meta_p.read_text())
            if old.get("sha256") == digest and old.get("render") == render:
                return {"doc_id": pdf_p.name, "skipped": True, "n_pages": old["n_pages"]}
        except (OSError, json.JSONDecodeError):
            pass
    t0 = time.time()
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "img").mkdir(parents=True)
    texts, pages = [], []
    with pymupdf.open(pdf_p) as doc:
        for i, page in enumerate(doc):
            n = i + 1
            text = page.get_text("text", sort=True).replace("\x00", "")
            w_pt, h_pt = page.rect.width, page.rect.height
            z = render_zoom(w_pt, h_pt, render)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=render["jpeg_quality"])
            (tmp / "img" / f"{n}.jpg").write_bytes(buf.getvalue())
            texts.append(text)
            pages.append({"page": n, "w_pt": round(w_pt, 1), "h_pt": round(h_pt, 1), "dpi": round(72 * z, 1),
                          "img_w": pix.width, "img_h": pix.height, "img_kb": round(len(buf.getvalue()) / 1024, 1),
                          "vis_tokens": vision_tokens(pix.width, pix.height, render["token_px"]),
                          "text_chars": len(text.strip())})
    (tmp / "text.json").write_text(json.dumps(texts, ensure_ascii=False))
    meta = {"doc_id": pdf_p.name, "sha256": digest, "render": render, "pymupdf": pymupdf.VersionBind,
            "n_pages": len(pages), "pages": pages,
            "seconds": round(time.time() - t0, 2)}
    (tmp / "meta.json").write_text(json.dumps(meta))
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return {"doc_id": pdf_p.name, "skipped": False, "n_pages": len(pages), "seconds": meta["seconds"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and pre-render MMLongBench-Doc for rci.tasks.mmlongdoc.")
    ap.add_argument("--raw", default=None, help="download directory (default: $RCI_DATA/raw/mmlongbench)")
    ap.add_argument("--no-fetch", action="store_true", help="use the files already under --raw")
    ap.add_argument("--samples", default="", help="samples.json path (default: <raw>/samples.json or <raw>/data/)")
    ap.add_argument("--allow-other-samples", action="store_true")
    ap.add_argument("--out", default=None, help="default: $RCI_DATA/mmlongdoc")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--docs", default="", help="comma-separated doc_ids (default: every doc in samples.json)")
    a = ap.parse_args()
    raw = Path(a.raw or paths.data_dir("raw", "mmlongbench"))
    out = Path(a.out or paths.data_dir("mmlongdoc"))
    out.mkdir(parents=True, exist_ok=True)
    if not a.no_fetch:
        fetch(raw, out)
    samples_p, docs_dir = (Path(a.samples) if a.samples else find_samples(raw)), find_documents(raw)
    digest = sha256(samples_p)
    if digest != SAMPLES_SHA256 and not a.allow_other_samples:
        sys.exit(f"[prepare] {samples_p} sha256 {digest[:12]} != pinned {SAMPLES_SHA256[:12]} (pinned upstream); "
                 "pass --allow-other-samples to use it anyway")
    samples = load_samples(samples_p)
    if len(samples) != N_SAMPLES:
        print(f"[prepare] WARNING: {len(samples)} questions in {samples_p}, expected {N_SAMPLES}")
    doc_ids = sorted({s["doc_id"] for s in samples})
    missing = [d for d in doc_ids if not (docs_dir / d).exists()]
    if missing:
        sys.exit(f"[prepare] {len(missing)} documents missing, e.g. {missing[:3]}")
    if a.docs:
        doc_ids = [d for d in a.docs.split(",") if d]
    (out / "docs").mkdir(parents=True, exist_ok=True)
    tmp = out / "samples.jsonl.tmp"
    with open(tmp, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(tmp, out / "samples.jsonl")

    t0, results, failures = time.time(), [], {}
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(prepare_doc, str(docs_dir / d), str(out / "docs" / doc_stem(d)), RENDER): d
                for d in doc_ids}
        for k, fu in enumerate(as_completed(futs), 1):
            d = futs[fu]
            try:
                results.append(fu.result())
            except Exception as e:
                failures[d] = f"{type(e).__name__}: {e}"
            if k % 10 == 0 or k == len(futs):
                print(f"[prepare] {k}/{len(futs)} docs, {time.time() - t0:.0f}s", flush=True)

    pages = n_text = 0
    toks, docs, libs = [], {}, {}
    for d in sorted({s["doc_id"] for s in samples}):
        mp = out / "docs" / doc_stem(d) / "meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        lib = m.get("pymupdf", "unrecorded")
        libs[lib] = libs.get(lib, 0) + 1
        docs[d] = {"n_pages": m["n_pages"], "sha256": m["sha256"],
                   "text_pages": sum(p["text_chars"] >= 50 for p in m["pages"])}
        pages += m["n_pages"]
        n_text += docs[d]["text_pages"]
        toks += [p["vis_tokens"] for p in m["pages"]]
    toks.sort()
    q = lambda f: toks[min(len(toks) - 1, int(f * len(toks)))] if toks else None
    integrity = raw / "INTEGRITY.json"
    manifest = {"samples_json": str(samples_p), "samples_sha256": digest, "n_questions": len(samples),
                "n_docs": len(docs), "n_docs_expected": len({s["doc_id"] for s in samples}), "n_pages": pages,
                "pages_with_text_ge50": n_text, "render": RENDER, "pymupdf_versions": libs,
                "vis_tokens": {"mean": round(sum(toks) / len(toks), 1) if toks else None, "median": q(0.5),
                               "p95": q(0.95), "max": toks[-1] if toks else None},
                "failures": failures, "seconds_this_run": round(time.time() - t0, 1),
                "integrity": json.loads(integrity.read_text()) if integrity.exists() else None, "docs": docs}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "docs"}, indent=1))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
