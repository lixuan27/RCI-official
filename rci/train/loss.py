"""Training items, batching and the clipped policy-gradient loss."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class TrainItem:
    ids: list[int]
    start: int
    behavior: list[float]
    adv: float
    weight: float = 1.0
    question: str = ""
    spans: list | None = None

    @property
    def n(self) -> int:
        return sum(n for _, n in self.spans) if self.spans else len(self.ids) - self.start

    def positions(self) -> list[int]:
        """Positions whose tokens are trained (each predicted from the position before it)."""
        if not self.spans:
            return list(range(self.start, len(self.ids)))
        return [p for s, n in self.spans for p in range(s, s + n)]


def items_from_samples(samples, l_max: int) -> tuple[list[TrainItem], dict]:
    """Drop zero-variance question groups, over-long sequences and malformed samples."""
    by_q = defaultdict(list)
    for s in samples:
        by_q[s.question].append(s)
    items, st = [], {"n_samples": len(samples), "dropped_sigma0": 0, "dropped_len": 0, "dropped_bad": 0,
                     "n_groups": len(by_q), "groups_sigma0": 0}
    for q, ss in by_q.items():
        rs = [s.reward for s in ss]
        if max(rs) - min(rs) <= 1e-12:
            st["groups_sigma0"] += 1
            st["dropped_sigma0"] += len(ss)
            continue
        for s in ss:
            ids = list(s.prompt_ids) + list(s.completion_ids)
            spans = [list(x) for x in s.spans] if s.spans else None
            n_lp = sum(n for _, n in spans) if spans else len(s.completion_ids)
            ok_spans = spans is None or (
                all(0 < a and n > 0 for a, n in spans)
                and all(a1 + n1 <= a2 for (a1, n1), (a2, _) in zip(spans, spans[1:]))
                and spans[-1] == [len(s.prompt_ids), len(s.completion_ids)])
            if not s.completion_ids or not s.prompt_ids or len(s.logprobs) != n_lp or not ok_spans:
                st["dropped_bad"] += 1
                continue
            if len(ids) > l_max:
                st["dropped_len"] += 1
                continue
            items.append(TrainItem(ids, spans[0][0] if spans else len(s.prompt_ids), list(s.logprobs),
                                   float(s.advantage), float(s.meta.get("weight", 1.0)), q, spans))
    return items, st


def padded_len(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def pad_right(ids: list[int], multiple: int, pad_id: int = 0) -> list[int]:
    """Right padding is exact for a causal model: no position attends to (or convolves over) later ones."""
    return ids + [pad_id] * (padded_len(len(ids), multiple) - len(ids))


def split_minibatches(items: list[TrainItem], m: int, seed: int) -> list[list[TrainItem]]:
    import random

    m = max(1, min(m, len(items)))
    if m == 1:
        return [list(items)]
    order = list(items)
    random.Random(seed).shuffle(order)
    parts: list[list[TrainItem]] = [[] for _ in range(m)]
    loads = [0] * m
    for it in order:
        j = min(range(m), key=lambda i: (loads[i], len(parts[i])))
        parts[j].append(it)
        loads[j] += it.n
    return parts


def balance(items: list[TrainItem], world: int, multiple: int) -> list[list[TrainItem | None]]:
    loads = [0] * world
    shards: list[list[TrainItem | None]] = [[] for _ in range(world)]
    for it in sorted(items, key=lambda x: -len(x.ids)):
        r = min(range(world), key=lambda i: (loads[i], len(shards[i])))
        shards[r].append(it)
        loads[r] += padded_len(len(it.ids), multiple)
    m = max((len(s) for s in shards), default=0)
    for s in shards:
        s.extend([None] * (m - len(s)))
    return shards


def token_pg(lp, mu, adv, eps_lo: float, eps_hi: float, tis_c: float, lp_old=None):
    """Token-level clipped PG with truncated importance sampling against the vLLM behaviour policy (decoupled PPO)."""
    import torch

    lp_old = lp.detach() if lp_old is None else lp_old
    ratio = torch.exp(lp - lp_old)
    is_w = torch.exp(lp_old - mu)
    w = torch.clamp(is_w, max=tis_c)
    pg = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi) * adv)
    d = mu - lp_old
    r = ratio.detach()
    clipped = (r > 1 + eps_hi) & (adv > 0) | (r < 1 - eps_lo) & (adv < 0)
    stats = {"k3_sum": (torch.exp(-d) - 1 + d).sum(), "tis_clipped": (is_w > tis_c).sum(), "ppo_clipped": clipped.sum()}
    return -(w * pg), stats
