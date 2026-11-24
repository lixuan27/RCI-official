"""Incremental BM25 index over a growing message collection."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

try:
    import Stemmer

    _STEMMER = Stemmer.Stemmer("english")

    def _stem(tokens: list[str]) -> list[str]:
        return _STEMMER.stemWords(tokens)
except Exception:
    def _stem(tokens: list[str]) -> list[str]:
        return tokens

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with".split()
)


def tokenize(text: str, bigrams: bool = False) -> list[str]:
    toks = [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    toks = _stem(toks)
    if bigrams and len(toks) > 1:
        toks = toks + [f"{a}__{b}" for a, b in zip(toks, toks[1:])]
    return toks


class IncrementalBM25:
    """BM25 with Lucene-style idf, k1=1.5 and b=0.75 by default."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, bigrams: bool = False):
        self.k1 = k1
        self.b = b
        self.bigrams = bigrams
        self.doc_len: list[int] = []
        self.tf: list[Counter] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.total_len = 0

    def __len__(self) -> int:
        return len(self.doc_len)

    def add(self, text: str) -> int:
        toks = tokenize(text, self.bigrams)
        tf = Counter(toks)
        doc_id = len(self.doc_len)
        self.doc_len.append(len(toks))
        self.tf.append(tf)
        self.total_len += len(toks)
        for term in tf:
            self.postings[term].append(doc_id)
        return doc_id

    def scores(self, query: str, candidates: Iterable[int] | None = None) -> dict[int, float]:
        """Return {doc_id: score} for documents with a positive score."""
        n = len(self.doc_len)
        if n == 0:
            return {}
        allowed = set(candidates) if candidates is not None else None
        avgdl = self.total_len / n if self.total_len else 1.0
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query, self.bigrams)):
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for d in plist:
                if allowed is not None and d not in allowed:
                    continue
                f = self.tf[d][term]
                denom = f + self.k1 * (1.0 - self.b + self.b * self.doc_len[d] / avgdl)
                out[d] += idf * f * (self.k1 + 1.0) / denom
        return {d: s for d, s in out.items() if s > 0}
