"""Candidate snippet pairs for contradiction injection."""
from __future__ import annotations

import random
import re
from collections import defaultdict

WORD_RE = re.compile(r"\d+(?:\.\d+)+|\w+")
_DIGIT = re.compile(r"\d")


def terms(text: str) -> set[str]:
    out = set()
    for w in WORD_RE.findall(text.lower()):
        if len(w) >= 4 or _DIGIT.search(w):
            out.add(w)
    return out


def body(snippet_text: str) -> str:
    """Strip the header line (and the blank line after it)."""
    i = snippet_text.find("\n")
    return snippet_text[i + 1:] if i >= 0 else ""


def mine(bodies: list[str], max_df: int = 8, min_shared: int = 2, seed: int = 0):
    postings: dict[str, list[int]] = defaultdict(list)
    for i, b in enumerate(bodies):
        for t in terms(b):
            postings[t].append(i)
    rare = {t: ix for t, ix in postings.items() if 2 <= len(ix) <= max_df}
    shared: dict[tuple[int, int], list[str]] = defaultdict(list)
    for t in sorted(rare):
        ix = rare[t]
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                shared[(ix[a], ix[b])].append(t)
    pairs = [(i, j, ts) for (i, j), ts in shared.items() if len(ts) >= min_shared]
    pairs.sort(key=lambda x: (x[0], x[1]))
    random.Random(seed).shuffle(pairs)
    pairs.sort(key=lambda x: -len(x[2]))
    n_rare_total = sum(1 for ix in postings.values() if len(ix) <= max_df)
    return pairs, {"n_terms": len(postings), "n_rare_df_le_max": n_rare_total, "n_rare_shared": len(rare)}
