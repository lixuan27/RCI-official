"""Hash orders, a deterministic random stream and small file helpers shared by the chains."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hkey(*parts) -> str:
    """sha256 of the parts joined by '/': the key of every hash order."""
    return sha("/".join(str(p) for p in parts))


def order(items, salt: str) -> list:
    """The items sorted by sha256(salt + item): a fixed order no reading can change."""
    return sorted(items, key=lambda x: sha(salt + str(x)))


class DRng:
    """SHA-256 in counter mode: the same stream on every Python version."""

    def __init__(self, *key):
        self._key = "|".join(str(k) for k in key).encode("utf-8")
        self._i = 0

    def _u64(self) -> int:
        h = hashlib.sha256(self._key + b"#" + str(self._i).encode("ascii")).digest()
        self._i += 1
        return int.from_bytes(h[:8], "big")

    def below(self, n: int) -> int:
        if n <= 0:
            raise ValueError("DRng.below needs n > 0")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = self._u64()
            if x < limit:
                return x % n

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def read_jsonl(path) -> list:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return p
