"""Paired gains on confirmation tasks and the release rules of the three chains.

A round compares the selected candidate with the current generation on confirmation tasks no earlier round has used,
paired by task: d is the mean per-task difference (candidate minus current generation) and SE its standard error.

    PlanBench    release when d >= 1.645 SE and d > 0
    HiddenBench  release when d > 0.026 (plus 0 x SE), and the control-task floor holds (a health line)
    Hanabi       release when d - 1.0 SE > 0 on the challenge score (SE: bootstrap over deals)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


class StatsError(ValueError):
    pass


def paired(cand: dict, inc: dict, min_share: float = 0.0) -> dict:
    """Mean and standard error of cand[k] - inc[k] over the tasks both hold.

    min_share: refuse when fewer than this share of the larger side's tasks are shared."""
    keys = sorted(set(cand) & set(inc))
    most = max(len(cand), len(inc), 1)
    if len(keys) < math.ceil(min_share * most):
        raise StatsError(f"{len(keys)} shared tasks of {most}: below the required share {min_share}")
    d = [float(cand[k]) - float(inc[k]) for k in keys]
    n = len(d)
    if n < 2:
        return {"n": n, "d": float("nan"), "se": float("nan")}
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return {"n": n, "d": m, "se": sd / math.sqrt(n)}


def bootstrap_paired(cand: dict, inc: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """The same mean difference with a bootstrap SE: tasks resampled with replacement, n_boot draws."""
    keys = sorted(set(cand) & set(inc))
    if set(cand) != set(inc):
        raise StatsError("the two arms are not on one task list")
    d = np.array([float(cand[k]) - float(inc[k]) for k in keys], dtype=np.float64)
    n = len(d)
    if n < 2:
        return {"n": n, "d": float("nan"), "se": float("nan")}
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return {"n": n, "d": float(d.mean()), "se": float(np.std(draws, ddof=1)), "n_boot": n_boot}


@dataclass(frozen=True)
class ReleaseRule:
    """Release when d - k SE > margin (>= when not strict) and, with positive, d > 0."""
    k: float = 0.0
    margin: float = 0.0
    strict: bool = True
    positive: bool = False

    @classmethod
    def of(cls, cfg: dict) -> "ReleaseRule":
        return cls(float(cfg.get("k", 0.0)), float(cfg.get("margin", 0.0)), bool(cfg.get("strict", True)),
                   bool(cfg.get("positive", False)))

    def line(self, se: float) -> float:
        return self.margin + self.k * se

    def passes(self, d: float, se: float) -> bool:
        if d != d or (self.k and se != se):
            return False
        lhs = d - self.k * se
        ok = lhs > self.margin if self.strict else lhs >= self.margin
        return bool(ok and (d > 0 or not self.positive))


PLANBENCH = ReleaseRule(k=1.645, margin=0.0, strict=False, positive=True)
HIDDENBENCH = ReleaseRule(k=0.0, margin=0.026, strict=True, positive=False)
HANABI = ReleaseRule(k=1.0, margin=0.0, strict=True, positive=False)


def significantly_worse(d: float, se: float, k: float) -> bool:
    """d <= -k SE: the mirror of the release line (a NaN difference counts as worse)."""
    return not (d == d and se == se and d > -k * se)


def lr_doubling(base_lr: float, rounds: list, max_doublings: int, k: float, halve_after_unhealthy: int = 0) -> float:
    """PlanBench: every flat round whose candidate was not significantly worse doubles the rate of the rounds after it
    (at most max_doublings times); halve_after_unhealthy unhealthy rounds in a row halve it once for good."""
    held, run, halved = 0, 0, False
    for x in rounds:
        lab = x.get("label")
        if lab == "flat":
            c = x.get("confirm") or {}
            if not significantly_worse(float(c.get("d", float("nan"))), float(c.get("se", float("nan"))), k):
                held += 1
        run = run + 1 if lab == "unhealthy" else 0
        if halve_after_unhealthy and run >= halve_after_unhealthy:
            halved = True
    return base_lr * 2 ** min(held, max_doublings) * (0.5 if halved else 1.0)


def lr_alpha_pairs(base_lr: float, winners: list, bounds) -> float:
    """Hanabi: two consecutive releases at alpha 2 double the rate, two at alpha 1/2 halve it, inside bounds."""
    lr, i = base_lr, 1
    while i < len(winners):
        a, b = Fraction(str(winners[i - 1])), Fraction(str(winners[i]))
        if a == b == 2:
            lr, i = min(bounds[1], lr * 2), i + 2
        elif a == b == Fraction(1, 2):
            lr, i = max(bounds[0], lr / 2), i + 2
        else:
            i += 1
    return lr
