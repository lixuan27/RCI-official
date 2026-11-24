"""Paragraph splitting and greedy snippet packing."""
from __future__ import annotations

from dataclasses import dataclass

HEADER = "File: {path}"
SEP_TOKENS = 1


def split_paragraphs(text: str) -> list[str]:
    paras: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.strip():
            cur.append(line.rstrip())
        elif cur:
            paras.append("\n".join(cur))
            cur = []
    if cur:
        paras.append("\n".join(cur))
    return paras


@dataclass
class Packed:
    paragraphs: list[str]
    body_tokens: int


def pack(paras: list[str], ntoks: list[int], target: int, mode: str = "after") -> list[Packed]:
    """Greedy packing; returns body groups (header not included)."""
    out: list[Packed] = []
    cur: list[str] = []
    cur_t = 0
    for p, t in zip(paras, ntoks):
        add = t + (SEP_TOKENS if cur else 0)
        if mode == "before" and cur and cur_t + add > target:
            out.append(Packed(cur, cur_t))
            cur, cur_t, add = [], 0, t
        cur.append(p)
        cur_t += add
        if mode == "after" and cur_t >= target:
            out.append(Packed(cur, cur_t))
            cur, cur_t = [], 0
    if cur:
        out.append(Packed(cur, cur_t))
    return out


def render(path: str, group: Packed) -> str:
    return HEADER.format(path=path) + "\n\n" + "\n\n".join(group.paragraphs)
