"""Contradiction-audit corpus build: filter -> paragraph snippets -> candidate pairs -> stats."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from rci import paths
from rci.tasks.audit.build import candidates as C
from rci.tasks.audit.build import filters as F
from rci.tasks.audit.build import snippets as S
from rci.tasks.audit.build.projects import Project, get

REFERENCE = {
    "cpython": (507, 2806), "docker": (930, 2074), "godot": (1586, 4512), "jax": (233, 920),
    "kubernetes": (1630, 3115), "linux": (3433, 8319), "llvm": (1229, 2663), "numpy": (199, 573),
    "pandas": (78, 392), "pytorch": (245, 597), "ray": (534, 1347), "scipy": (339, 635),
    "sklearn": (102, 475), "sqlalchemy": (152, 583), "sympy": (317, 732), "transformers": (726, 1507),
}

_ENC = None


def enc():
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("o200k_base")
    return _ENC


def _write_jsonl(path: Path, rows) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def read_text(path: Path) -> str:
    b = path.read_bytes()
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("utf-8", errors="replace")


def collect(p: Project, src: Path) -> list[tuple[str, str]]:
    """(repo-relative path, root-relative path) for every doc file, sorted by path."""
    repo = src / p.split / p.name
    seen = {}
    for root in p.roots:
        base = repo / root if root else repo
        if not base.is_dir():
            continue
        for rr in F.list_doc_files(str(base), p.exts):
            rel = f"{root.strip('/')}/{rr}" if root.strip("/") else rr
            seen.setdefault(rel, rr)
    return sorted(seen.items())


def build_project(p: Project, src: Path, out: Path, target: int, mode: str, max_df: int,
                  min_shared: int, seed: int, top: int, threads: int, do_cands: bool) -> dict:
    t0 = time.time()
    repo = src / p.split / p.name
    odir = out / p.split / p.name
    odir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((repo / ".audit_fetch.json").read_text())
    paths = collect(p, src)
    symlinks = {rel for rel, _ in paths if (repo / rel).is_symlink()}
    texts = {rel: read_text(repo / rel) for rel, _ in paths if rel not in symlinks}
    transcluded = F.resolve_transcluded(texts, p.exts)

    rows, kept = [], []
    for rel, rootrel in paths:
        if rel in symlinks:
            rows.append({"file": rel, "bytes": 0, "decision": "excluded", "rule": "symlink", "category": "include",
                         "includer": os.readlink(repo / rel)})
            continue
        text = texts[rel]
        row = {"file": rel, "bytes": len(text.encode("utf-8"))}
        if rel in transcluded:
            row.update(decision="excluded", rule="transcluded", category="include", includer=transcluded[rel])
        else:
            r = F.classify(p.name, rel, rootrel, text)
            if r is not None:
                row.update(decision="excluded", rule=r.rid, category=r.category)
        rows.append(row)
        if "decision" not in row:
            kept.append(row)

    paras_by_file = {row["file"]: S.split_paragraphs(texts[row["file"]]) for row in kept}
    flat = [para for row in kept for para in paras_by_file[row["file"]]]
    ntok_flat = [len(x) for x in enc().encode_ordinary_batch(flat, num_threads=threads)] if flat else []

    snippets, k = [], 0
    n_before = 0
    cursor = 0
    for row in kept:
        paras = paras_by_file[row["file"]]
        nt = ntok_flat[cursor:cursor + len(paras)]
        cursor += len(paras)
        row["n_paragraphs"] = len(paras)
        row["max_para_tokens"] = max(nt) if nt else 0
        if not paras:
            row.update(decision="skipped", rule="empty", category="empty")
            continue
        if max(nt) > 2 * target:
            row.update(decision="skipped", rule="long_paragraph", category="long_paragraph")
            continue
        groups = S.pack(paras, nt, target, mode)
        n_before += len(S.pack(paras, nt, target, "before" if mode == "after" else "after"))
        row.update(decision="kept", n_snippets=len(groups), body_tokens=sum(g.body_tokens for g in groups))
        for j, g in enumerate(groups):
            snippets.append({"id": f"{p.name}/{k:05d}", "idx": k, "file": row["file"], "part": j,
                             "n_parts": len(groups), "whole_file": len(groups) == 1,
                             "text": S.render(row["file"], g)})
            k += 1
    if snippets:
        full = enc().encode_ordinary_batch([s["text"] for s in snippets], num_threads=threads)
        for s, toks in zip(snippets, full):
            s["n_tokens"] = len(toks)
    _write_jsonl(odir / "files.jsonl", rows)
    _write_jsonl(odir / "snippets.jsonl", snippets)

    cstats = {}
    if do_cands:
        pairs, cstats = C.mine([C.body(s["text"]) for s in snippets], max_df, min_shared, seed)
        n_same = sum(1 for i, j, _ in pairs if snippets[i]["file"] == snippets[j]["file"])
        cstats.update(n_pairs=len(pairs), n_pairs_same_file=n_same, n_pairs_cross_file=len(pairs) - n_same)
        hist = {}
        for _, _, ts in pairs:
            hist[len(ts)] = hist.get(len(ts), 0) + 1
        cstats["shared_hist"] = {str(kk): hist[kk] for kk in sorted(hist)}

        def cand_rows():
            for r, (i, j, ts) in enumerate(pairs[:top]):
                yield {"rank": r, "a": snippets[i]["id"], "b": snippets[j]["id"], "shared": len(ts),
                       "same_file": snippets[i]["file"] == snippets[j]["file"], "terms": ts[:16]}
        _write_jsonl(odir / "candidates.jsonl", cand_rows())

    by_cat: dict[str, int] = {}
    for row in rows:
        if row.get("decision") != "kept":
            by_cat[row.get("category", "?")] = by_cat.get(row.get("category", "?"), 0) + 1
    ntoks = [s["n_tokens"] for s in snippets]
    st = {
        "project": p.name, "split": p.split, "repo": p.repo, "sha": meta["sha"], "commit_date": meta["commit_date"],
        "roots": list(p.roots), "n_doc_files": len(rows), "excluded_by_category": by_cat,
        "n_files": sum(1 for r in rows if r.get("decision") == "kept"),
        "n_snippets": len(snippets), "n_snippets_alt_mode": n_before, "mode": mode, "target": target,
        "n_whole_file_snippets": sum(1 for s in snippets if s["whole_file"]),
        "tokens_total": sum(ntoks), "tokens_mean": round(sum(ntoks) / max(1, len(ntoks)), 1),
        "tokens_max": max(ntoks) if ntoks else 0, "candidates": cstats,
        "seconds": round(time.time() - t0, 1),
    }
    if p.name in REFERENCE:
        st["ref_files"], st["ref_snippets"] = REFERENCE[p.name]
    _write_json(odir / "stats.json", st)
    return st


def _worker(args):
    p, kw = args
    try:
        return build_project(p, **kw)
    except Exception as e:
        import traceback
        return {"project": p.name, "split": p.split, "error": repr(e), "tb": traceback.format_exc()}


def table(stats: list[dict], split: str) -> str:
    ref = split == "eval"
    hdr = "| project | doc files | excluded | skipped | files | snippets | alt-mode snippets |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    if ref:
        hdr += " ref files | ref snippets | files/ref | snippets/ref |"
        sep += "---:|---:|---:|---:|"
    lines = [hdr, sep]
    tot = dict(d=0, e=0, s=0, f=0, n=0, a=0, rf=0, rs=0)
    for st in sorted(stats, key=lambda x: x["project"]):
        exc = sum(v for c, v in st["excluded_by_category"].items() if c not in ("empty", "long_paragraph"))
        skp = st["excluded_by_category"].get("empty", 0) + st["excluded_by_category"].get("long_paragraph", 0)
        line = f"| {st['project']} | {st['n_doc_files']} | {exc} | {skp} | {st['n_files']} | {st['n_snippets']} | {st['n_snippets_alt_mode']} |"
        tot["d"] += st["n_doc_files"]; tot["e"] += exc; tot["s"] += skp
        tot["f"] += st["n_files"]; tot["n"] += st["n_snippets"]; tot["a"] += st["n_snippets_alt_mode"]
        if ref:
            rf, rs = st.get("ref_files", 0), st.get("ref_snippets", 0)
            tot["rf"] += rf; tot["rs"] += rs
            line += f" {rf} | {rs} | {st['n_files'] / max(1, rf):.2f} | {st['n_snippets'] / max(1, rs):.2f} |"
        lines.append(line)
    line = f"| **total** | {tot['d']} | {tot['e']} | {tot['s']} | {tot['f']} | {tot['n']} | {tot['a']} |"
    if ref:
        line += f" {tot['rf']} | {tot['rs']} | {tot['f'] / max(1, tot['rf']):.2f} | {tot['n'] / max(1, tot['rs']):.2f} |"
    lines.append(line)
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.audit.build.build",
                                 description="Build snippets and candidate pairs from the fetched audit corpora.")
    ap.add_argument("--src", type=Path, default=None, help="fetched corpora (default: $RCI_DATA/audit/corpora)")
    ap.add_argument("--out", type=Path, default=None, help="default: $RCI_DATA/audit")
    ap.add_argument("--split", default="eval,train")
    ap.add_argument("--only", default="")
    ap.add_argument("--target", type=int, default=1024)
    ap.add_argument("--mode", choices=("after", "before"), default="after")
    ap.add_argument("--max-df", type=int, default=8)
    ap.add_argument("--min-shared", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-candidates", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args(argv)
    a.src = a.src or paths.data_dir("audit", "corpora")
    a.out = a.out or paths.data_dir("audit")
    names = [s for s in a.only.split(",") if s] or None
    projs = [p for p in get(names) if p.split in a.split.split(",")]
    projs = [p for p in projs if (a.src / p.split / p.name / ".audit_fetch.json").exists()]
    if a.skip_done:
        projs = [p for p in projs if not (a.out / p.split / p.name / "stats.json").exists()]
    kw = dict(src=a.src, out=a.out, target=a.target, mode=a.mode, max_df=a.max_df, min_shared=a.min_shared,
              seed=a.seed, top=a.top, threads=a.threads, do_cands=not a.no_candidates)
    projs.sort(key=lambda p: -sum(1 for _ in collect(p, a.src)))
    results = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for st in pool.imap_unordered(_worker, [(p, kw) for p in projs]):
            results.append(st)
            msg = st.get("error") or f"files={st['n_files']} snippets={st['n_snippets']} pairs={st['candidates'].get('n_pairs')} {st['seconds']}s"
            print(f"[{st['split']}/{st['project']}] {msg}", flush=True)
            if "tb" in st:
                print(st["tb"], flush=True)
    bad = [r for r in results if "error" in r]
    for split in a.split.split(","):
        sts = []
        for p in get(None, split):
            f = a.out / split / p.name / "stats.json"
            if f.exists():
                sts.append(json.loads(f.read_text()))
        if sts:
            _write_json(a.out / f"stats_{split}.json", sts)
            (a.out / f"stats_{split}.md").write_text(table(sts, split))
            print(table(sts, split), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
