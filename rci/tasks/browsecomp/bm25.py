"""BM25 index over the BrowseComp-Plus corpus (bm25s), with a Lucene-like analyzer."""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import sys
import time

from rci import paths

LUCENE_EN_STOPWORDS = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
    "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with",
]
_STOP = frozenset(LUCENE_EN_STOPWORDS)
_POSSESSIVE = re.compile(r"['\u2019\uff07]s\b")
_TOKEN = re.compile(r"\w+", re.UNICODE)
K1, B = 0.9, 0.4


@functools.lru_cache(maxsize=None)
def _stemmer():
    import Stemmer
    return Stemmer.Stemmer("porter")


class Analyzer:
    """Lucene-EnglishAnalyzer-like analyzer with a per-process stem cache."""

    def __init__(self):
        self.stem = _stemmer().stemWord
        self.cache: dict[str, str] = {}

    def __call__(self, text: str) -> list[str]:
        out = []
        cache = self.cache
        for tok in _TOKEN.findall(_POSSESSIVE.sub("", text.lower())):
            if tok in _STOP:
                continue
            s = cache.get(tok)
            if s is None:
                s = self.stem(tok)
                cache[tok] = s
            out.append(s)
        return out


def load_corpus(root: str):
    docids, texts = [], []
    with open(f"{root}/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docids.append(r["docid"])
            texts.append(r["text"])
    return docids, texts


def cmd_build(root: str) -> None:
    import bm25s
    t0 = time.time()
    docids, texts = load_corpus(root)
    print(f"[build] loaded {len(docids)} docs in {time.time()-t0:.0f}s", flush=True)
    an = Analyzer()
    vocab: dict[str, int] = {}
    ids_all = []
    n_tok = 0
    for i, t in enumerate(texts):
        toks = an(t)
        n_tok += len(toks)
        ids_all.append([vocab.setdefault(w, len(vocab)) for w in toks])
        if i % 10000 == 0:
            print(f"[build] tokenized {i} docs, vocab={len(vocab)}, tokens={n_tok}, {time.time()-t0:.0f}s", flush=True)
    del texts
    print(f"[build] tokenized all: vocab={len(vocab)} tokens={n_tok} ({time.time()-t0:.0f}s)", flush=True)
    retriever = bm25s.BM25(k1=K1, b=B, method="lucene")
    retriever.index((ids_all, vocab), show_progress=False)
    out = f"{root}/index/bm25s"
    os.makedirs(out, exist_ok=True)
    retriever.save(out)
    with open(f"{out}/docids.json", "w") as f:
        json.dump(docids, f)
    with open(f"{out}/analyzer.json", "w") as f:
        json.dump({"lowercase": True, "possessive_regex": _POSSESSIVE.pattern,
                   "token_regex": _TOKEN.pattern, "stopwords": LUCENE_EN_STOPWORDS,
                   "stemmer": "PyStemmer porter", "k1": K1, "b": B, "method": "lucene",
                   "bm25s_version": bm25s.__version__, "n_docs": len(docids),
                   "n_tokens": n_tok, "vocab_size": len(vocab),
                   "indexed_field": "corpus.jsonl:text (full, incl. front matter)"}, f, indent=1)
    print(f"[build] saved to {out} in {time.time()-t0:.0f}s", flush=True)


class Searcher:
    """Load the index once (memory-mapped), then search(query, k)."""

    def __init__(self, index_dir: str):
        import bm25s
        self.r = bm25s.BM25.load(index_dir, mmap=True, show_progress=False)
        with open(f"{index_dir}/docids.json") as f:
            self.docids = json.load(f)
        self.an = Analyzer()

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        import numpy as np
        toks = [t for t in self.an(query) if t in self.r.vocab_dict]
        if not toks:
            return []
        scores = self.r.get_scores(toks)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((top, -scores[top]))]
        return [(self.docids[i], float(scores[i])) for i in top if scores[i] > 0]


def _load_queries(root):
    with open(f"{root}/queries.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def cmd_eval(root: str) -> None:
    import numpy as np
    s = Searcher(f"{root}/index/bm25s")
    qs = _load_queries(root)
    ks = (5, 20, 100, 1000)
    agg = {f"{kind}_R@{k}": [] for kind in ("evidence", "gold") for k in ks}
    agg.update({"evidence_nDCG@10": [], "gold_nDCG@10": []})
    lat = []
    run_path = f"{root}/index/bm25s_run_top1000.trec"
    with open(run_path, "w") as run:
        for q in qs:
            t = time.perf_counter()
            hits = s.search(q["question"], k=1000)
            lat.append(time.perf_counter() - t)
            ranked = [d for d, _ in hits]
            for rank, (d, sc) in enumerate(hits, 1):
                run.write(f"{q['query_id']} Q0 {d} {rank} {sc:.4f} bm25s\n")
            for kind in ("evidence", "gold"):
                rel = set(q[f"{kind}_docids"])
                if not rel:
                    continue
                for k in ks:
                    agg[f"{kind}_R@{k}"].append(len(rel & set(ranked[:k])) / len(rel))
                dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
                idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(rel))))
                agg[f"{kind}_nDCG@10"].append(dcg / idcg)
    res = {k: round(100 * float(np.mean(v)), 2) for k, v in agg.items()}
    res["n_queries"] = len(qs)
    res["latency_ms_mean"] = round(1000 * float(np.mean(lat)), 1)
    res["latency_ms_p95"] = round(1000 * float(np.percentile(lat, 95)), 1)
    first100 = qs[:100]
    res["first100_evidence_R@20"] = round(100 * float(np.mean([
        len(set(q["evidence_docids"]) & set(d for d, _ in s.search(q["question"], 20))) / len(q["evidence_docids"])
        for q in first100])), 2)
    with open(f"{root}/index/bm25s_eval.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)


def cmd_smoke(root: str) -> None:
    t0 = time.time()
    s = Searcher(f"{root}/index/bm25s")
    print(f"[smoke] index loaded (mmap) in {time.time()-t0:.1f}s", flush=True)
    titles = {}
    with open(f"{root}/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            titles[r["docid"]] = (r["title"], r["url"])
    qs = _load_queries(root)
    probes = [("q0-question", qs[0]["question"], set(qs[0]["evidence_docids"])),
              ("short", "a novelist grew taller after finishing a book", set()),
              ("empty-after-analysis", "the of and", set())]
    for name, q, rel in probes:
        t = time.perf_counter()
        hits = s.search(q, k=20)
        dt = (time.perf_counter() - t) * 1000
        print(f"\n[smoke] {name}: {len(hits)} hits in {dt:.1f} ms; evidence hits in top20 = "
              f"{sum(d in rel for d, _ in hits)}/{len(rel)}")
        for rank, (d, sc) in enumerate(hits[:5]):
            print(f"  {rank:2d} {d:>7s} {sc:7.3f} {titles[d][0][:70]!r} {titles[d][1][:60]}")
    print("[smoke] OK", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.browsecomp.bm25",
                                 description="Build, evaluate or smoke-test the BrowseComp-Plus BM25 index.")
    ap.add_argument("cmd", choices=["build", "eval", "smoke"])
    ap.add_argument("--root", default=None, help="default: $RCI_DATA/browsecomp")
    a = ap.parse_args(argv)
    {"build": cmd_build, "eval": cmd_eval, "smoke": cmd_smoke}[a.cmd](a.root or str(paths.data_dir("browsecomp")))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
