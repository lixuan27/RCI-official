"""The scripted shortcut probe of DocSweep admission: search, open a few documents, answer."""
from __future__ import annotations

import json
import re

from rci.tasks.docsweep import normalise as N
from rci.tasks.docsweep.task import search_hits

MAX_OPEN = 11
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")


def probe_terms(example: dict) -> list[str]:
    stop = set(_stopwords())
    words = []
    for w in _WORD.findall(example["question"]):
        lw = w.lower().strip("'-")
        if len(lw) > 3 and lw not in stop and lw not in words:
            words.append(lw)
    t = N.domain_templates(example["domain"])
    syn = list(t["drop"]["cues"]) + list(t["level"]["base_cues"]) + list(t["level"]["offset_cues"])
    for vals in t["syn"].values():
        syn += list(vals)
    for s in syn:
        if s.lower() not in words:
            words.append(s.lower())
    return words


def _stopwords() -> list[str]:
    c = json.loads((N.TEMPLATES / "common.json").read_text(encoding="utf-8"))
    return [w.lower() for w in c["probe_stopwords"]]


def run_probe(example: dict, tokenizer, max_open: int = MAX_OPEN) -> tuple[list[str], list[int]]:
    """The probe's answer and the documents it opened (at most `max_open`, in hit order)."""
    opened: list[int] = []
    for term in probe_terms(example):
        for d, _, _, _ in search_hits(example, term, tokenizer):
            if d not in opened:
                opened.append(d)
            if len(opened) >= max_open:
                break
        if len(opened) >= max_open:
            break
    return N.solve(example, opened), opened


def shortcut_answer(example: dict, tokenizer) -> list[str]:
    return run_probe(example, tokenizer)[0]
