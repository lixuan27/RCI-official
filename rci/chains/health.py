"""Health lines: a candidate is selectable only when every line holds against the base model's own reading."""
from __future__ import annotations

from dataclasses import asdict, dataclass

EPS = 1e-9


@dataclass
class Line:
    name: str
    value: float | None
    limit: object
    ok: bool

    def to_json(self) -> dict:
        return asdict(self)


def at_most(name: str, value, limit) -> Line:
    return Line(name, value, limit, value is not None and value <= limit + EPS)


def at_least(name: str, value, limit) -> Line:
    return Line(name, value, limit, value is not None and value >= limit - EPS)


def within(name: str, value, lo, hi) -> Line:
    ok = value is not None and (lo is None or value >= lo - EPS) and (hi is None or value <= hi + EPS)
    return Line(name, value, [lo, hi], ok)


def ratio(x, base) -> float:
    return float("inf") if not base else float(x) / float(base)


def healthy(lines) -> bool:
    return all(x.ok for x in lines)


def failed(lines) -> list:
    return [x.name for x in lines if not x.ok]
