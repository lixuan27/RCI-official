"""Token counting shared by message limits and corpus snippetization (o200k_base)."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    enc = _encoder()
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= limit:
        return text
    return enc.decode(ids[:limit])
