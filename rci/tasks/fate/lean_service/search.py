"""`search_mathlib`: BM25 over the declarations of the pinned Mathlib environment."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z0-9]+|[A-Z]+|[^\x00-\x7f]")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9']*|\d+|[^\x00-\x7f\sA-Za-z0-9]")
STOP = {"the", "a", "an", "to", "for", "be", "by", "with", "on", "that", "this", "as", "at",
        "it", "we", "are", "u", "v", "w"}


def tokenize(text: str) -> list[str]:
    toks = []
    for w in _WORD.findall(text):
        if w.isascii() and len(w) > 1:
            low = w.lower()
            parts = [p.lower() for p in _CAMEL.findall(w)]
            toks.append(low)
            if len(parts) > 1:
                toks.extend(parts)
        else:
            toks.append(w.lower())
    return [t for t in toks if t not in STOP]


RERANK = 1.0


def name_tokens(name: str) -> list[str]:
    return tokenize(name.replace(".", " ").replace("_", " "))


def doc_tokens(d: dict) -> list[str]:
    name = d["name"]
    name_toks = name_tokens(name) + [name.lower()]
    sig = d.get("signature") or ""
    if sig.startswith(name):
        sig = sig[len(name):]
    doc = (d.get("doc") or "")[:400]
    return name_toks * 2 + tokenize(sig.replace("_", " ")) + tokenize(doc)


class MathlibIndex:
    def __init__(self, index_dir: str):
        import bm25s
        self.dir = Path(index_dir)
        self.retriever = bm25s.BM25.load(str(self.dir / "bm25"))
        self.vocab = json.loads((self.dir / "vocab.json").read_text())
        self.decls = [json.loads(l) for l in (self.dir / "decls.jsonl").read_text().splitlines()]
        self.by_name = {d["name"]: i for i, d in enumerate(self.decls)}

    def search(self, query: str, k: int = 8) -> list[dict]:
        k = max(1, min(int(k), 50))
        q = query.strip()
        hits: list[tuple[int, float]] = []
        if q in self.by_name:
            hits.append((self.by_name[q], float("inf")))
        toks = [t for t in tokenize(q.replace(".", " ").replace("_", " ")) + [q.lower()]
                if t in self.vocab]
        if toks:
            qset = set(toks)
            docs, scores = self.retriever.retrieve([toks], k=min(64, len(self.decls)),
                                                   show_progress=False)
            cand = []
            for i, s in zip(docs[0].tolist(), scores[0].tolist()):
                if s > 0 and all(i != h for h, _ in hits):
                    nt = set(name_tokens(self.decls[i]["name"]))
                    cover = len(nt & qset) / max(1, len(nt))
                    cand.append((int(i), float(s) * (1.0 + RERANK * cover)))
            hits += sorted(cand, key=lambda x: -x[1])
        out = []
        for i, s in hits[:k]:
            d = self.decls[i]
            out.append({"name": d["name"], "kind": d["kind"], "module": d["module"],
                        "signature": d["signature"], "doc": (d.get("doc") or "")[:300],
                        "score": None if s == float("inf") else round(s, 3)})
        return out


def format_results(results: list[dict], limit_chars: int = 6000) -> str:
    if not results:
        return "No matching Mathlib declarations."
    parts = []
    for r in results:
        s = f"{r['kind']} {r['signature']}\n  -- {r['module']}"
        if r["doc"]:
            s += "\n  -- " + " ".join(r["doc"].split())[:200]
        parts.append(s)
    text = "\n\n".join(parts)
    return text if len(text) <= limit_chars else text[:limit_chars] + "\n... [truncated]"


def build(decl_files: list[str], out_dir: str) -> dict:
    import bm25s
    t0 = time.time()
    decls, seen = [], set()
    for f in decl_files:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d["name"] in seen:
                continue
            seen.add(d["name"])
            decls.append(d)
    decls.sort(key=lambda d: d["name"])
    vocab: dict[str, int] = {}
    corpus_ids = []
    for d in decls:
        ids = []
        for t in doc_tokens(d):
            if t not in vocab:
                vocab[t] = len(vocab)
            ids.append(vocab[t])
        corpus_ids.append(ids)
    retriever = bm25s.BM25(k1=1.2, b=0.75)
    retriever.index(bm25s.tokenization.Tokenized(ids=corpus_ids, vocab=vocab),
                    show_progress=False)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    retriever.save(str(out / "bm25"))
    (out / "vocab.json").write_text(json.dumps(vocab))
    (out / "decls.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n"
                                             for d in decls))
    kinds: dict[str, int] = {}
    for d in decls:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    stats = {"n_decls": len(decls), "vocab": len(vocab), "kinds": kinds,
             "pp_failed": sum("<pp failed>" in d["signature"] for d in decls),
             "build_s": round(time.time() - t0, 1), "sources": decl_files}
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.fate.lean_service.search",
                                 description="Build or query the BM25 index over Mathlib declarations.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("decls", nargs="+")
    b.add_argument("--out", required=True)
    q = sub.add_parser("query")
    q.add_argument("index")
    q.add_argument("text", nargs="+")
    q.add_argument("-k", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build(a.decls, a.out), indent=2))
    else:
        idx = MathlibIndex(a.index)
        for t in a.text:
            t0 = time.time()
            res = idx.search(t, a.k)
            print(f"### {t!r} ({(time.time() - t0) * 1000:.1f} ms)")
            print(format_results(res))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
