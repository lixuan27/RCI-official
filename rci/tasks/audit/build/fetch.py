"""Fetch the documentation trees of the contradiction-audit corpora at pinned commits (stdlib only)."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rci import paths
from rci.tasks.audit.build.projects import DOC_EXTS, Project, get

GIT = ["git", "-c", "advice.detachedHead=false"]


def _run(args: list[str], cwd: Path | None, timeout: int, log) -> None:
    log.write(f"$ {' '.join(args)}\n")
    log.flush()
    r = subprocess.run(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"rc={r.returncode}: {' '.join(args[:6])}")


def patterns(p: Project) -> list[str]:
    pats = []
    for root in p.roots:
        base = f"/{root.strip('/')}/" if root.strip("/") else "/"
        for ext in p.exts:
            pats.append(f"{base}**/*{ext}")
    return pats


def count_docs(dest: Path, roots: tuple[str, ...], exts: tuple[str, ...] = DOC_EXTS) -> int:
    n = 0
    for root in roots:
        base = dest / root if root else dest
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            n += sum(1 for f in filenames if f.lower().endswith(exts))
    return n


def fetch_one(p: Project, out: Path, logdir: Path, timeout: int, attempts: int) -> dict:
    dest = out / p.split / p.name
    marker = dest / ".audit_fetch.json"
    if marker.exists():
        return json.loads(marker.read_text())
    logdir.mkdir(parents=True, exist_ok=True)
    last = None
    for k in range(attempts):
        with open(logdir / f"fetch_{p.name}.log", "a") as log:
            log.write(f"\n=== attempt {k} {dt.datetime.now().isoformat()} ===\n")
            try:
                if dest.exists():
                    shutil.rmtree(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                clone = GIT + ["clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse"]
                if p.ref:
                    clone += ["--branch", p.ref]
                _run(clone + [p.url, str(dest)], None, timeout, log)
                _run(GIT + ["sparse-checkout", "set", "--no-cone", *patterns(p)], dest, timeout, log)
                q = lambda *a: subprocess.run(["git", *a], cwd=dest, capture_output=True, text=True, timeout=60).stdout.strip()
                info = {
                    "name": p.name, "split": p.split, "repo": p.repo, "url": p.url,
                    "roots": list(p.roots), "branch": q("rev-parse", "--abbrev-ref", "HEAD"),
                    "sha": q("rev-parse", "HEAD"), "commit_date": q("log", "-1", "--format=%cI"),
                    "n_doc_files": count_docs(dest, p.roots, p.exts),
                    "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "seconds": round(time.time() - t0, 1),
                }
                if info["n_doc_files"] == 0:
                    raise RuntimeError("no doc files materialised")
                marker.write_text(json.dumps(info, indent=1))
                log.write(f"OK {json.dumps(info)}\n")
                return info
            except Exception as e:
                last = repr(e)
                log.write(f"FAIL {last}\n")
                time.sleep(10 * (k + 1))
    return {"name": p.name, "split": p.split, "repo": p.repo, "error": last}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.audit.build.fetch",
                                 description="Fetch the audit corpora at their pinned commits.")
    ap.add_argument("--out", type=Path, default=None, help="default: $RCI_DATA/audit/corpora")
    ap.add_argument("--logdir", type=Path, default=None, help="default: $RCI_LOGS/audit")
    ap.add_argument("--only", default="")
    ap.add_argument("--split", default=None)
    ap.add_argument("-j", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--attempts", type=int, default=3)
    a = ap.parse_args(argv)
    a.out = a.out or paths.data_dir("audit", "corpora")
    a.logdir = a.logdir or paths.logs_dir("audit")
    projs = get([s for s in a.only.split(",") if s] or None, a.split)
    results = []
    with cf.ThreadPoolExecutor(a.j) as ex:
        futs = {ex.submit(fetch_one, p, a.out, a.logdir, a.timeout, a.attempts): p for p in projs}
        for f in cf.as_completed(futs):
            r = f.result()
            results.append(r)
            print(json.dumps(r), flush=True)
    bad = [r for r in results if "error" in r]
    print(f"done: {len(results) - len(bad)} ok, {len(bad)} failed: {[r['name'] for r in bad]}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
