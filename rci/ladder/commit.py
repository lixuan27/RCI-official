"""Commitment rules: which single answer a group, or a set of independent members, commits to."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from rci.tasks.base import stake

SET_TASKS = {"audit"}
VERIFIED_TASKS = {"fate"}
UNPOOLED_TASKS = ("fate", "swe")
EXTRACT_FAILED = "Failed to extract"


def stated(final) -> float | None:
    """The confidence a submission states, or None."""
    if not isinstance(final, dict) or not final:
        return None
    return final.get("confidence")


def _verified(rec: dict, cand: list[int]) -> float | None:
    if rec.get("task") not in VERIFIED_TASKS:
        return None
    per = rec["score"]["per_agent"]
    return max((float(per[i]) for i in cand), default=0.0)


def _norm(s) -> str | None:
    if s is None:
        return None
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    s = s.strip(" .,:;!?\"'`")
    return s or None


def _candidates(rec: dict) -> list[int]:
    per = (rec.get("score") or {}).get("per_agent") or []
    out = []
    for i, (f, p) in enumerate(zip(rec.get("finals") or [], per)):
        if p is None or f.get("failing") or not f.get("final"):
            continue
        out.append(i)
    return out


def _vote_key(rec: dict, i: int):
    preds = (rec.get("score") or {}).get("preds")
    if preds is not None and i < len(preds):
        return None if preds[i] == EXTRACT_FAILED else _norm(preds[i])
    final = rec["finals"][i].get("final")
    return _norm(final.get("answer") if isinstance(final, dict) else final)


def top_confidence(rec: dict) -> float:
    """Score of the answer stated with the highest confidence, ties averaged."""
    per = rec["score"]["per_agent"]
    cand = _candidates(rec)
    if not cand:
        return 0.0
    v = _verified(rec, cand)
    if v is not None:
        return v
    conf = {i: stake(rec["finals"][i]["final"]) for i in cand}
    top = max(conf.values())
    tied = [i for i in cand if conf[i] == top]
    return sum(per[i] for i in tied) / len(tied)


def vote(rec: dict) -> float | None:
    """Score of the plurality answer, ties averaged; None for proofs and patches."""
    if rec.get("task") in UNPOOLED_TASKS:
        return None
    per = rec["score"]["per_agent"]
    cand = _candidates(rec)
    if not cand:
        return 0.0
    clusters: dict[str, list[int]] = defaultdict(list)
    for i in cand:
        k = _vote_key(rec, i)
        if k is not None:
            clusters[k].append(i)
    if not clusters:
        return 0.0
    m = max(len(v) for v in clusters.values())
    top = [v for v in clusters.values() if len(v) == m]
    return sum(sum(per[i] for i in v) / len(v) for v in top) / len(top)


def vote_stake(rec: dict) -> float:
    """Score of the answer with the most backers, ties broken by summed confidence and then averaged."""
    per = rec["score"]["per_agent"]
    cand = _candidates(rec)
    if not cand:
        return 0.0
    v = _verified(rec, cand)
    if v is not None:
        return v
    clusters: dict = defaultdict(list)
    for i in cand:
        k = None if rec.get("task") in UNPOOLED_TASKS else _vote_key(rec, i)
        clusters[k if k is not None else ("member", i)].append(i)

    def conf(i):
        return stake(rec["finals"][i]["final"])

    key = {k: (len(v), sum(conf(i) for i in v)) for k, v in clusters.items()}
    top = max(key.values())
    tied = [clusters[k] for k in clusters if key[k] == top]
    return sum(sum(per[i] for i in v) / len(v) for v in tied) / len(tied)


def _free(rec: dict) -> list[int]:
    cand = _candidates(rec)
    return [i for i in cand if not rec["finals"][i].get("forced")] or cand


def vote_free(rec: dict) -> float:
    """vote_stake over the members who submitted on their own (forced submissions cast no vote)."""
    if rec.get("task") in VERIFIED_TASKS:
        return vote_stake(rec)
    pool = set(_free(rec))
    return vote_stake({**rec, "finals": [f if i in pool else {**f, "final": None} for i, f in enumerate(rec["finals"])]})


def random_pick(rec: dict) -> float:
    """Mean score of the members' answers."""
    per = rec["score"]["per_agent"]
    cand = _candidates(rec)
    return sum(per[i] for i in cand) / len(cand) if cand else 0.0


def confidence_gap(rec: dict) -> float | None:
    """Mean confidence on correct answers minus mean confidence on wrong ones."""
    per = (rec.get("score") or {}).get("per_agent")
    if per is None:
        return None
    right, wrong = [], []
    for i in _candidates(rec):
        (right if per[i] > 0.5 else wrong).append(stake(rec["finals"][i]["final"]))
    return sum(right) / len(right) - sum(wrong) / len(wrong) if right and wrong else None


def metrics(rec: dict) -> dict:
    """Every commitment rule on one record; values are None for set tasks and ungraded records."""
    sc = rec.get("score") or {}
    keys = ("top_confidence", "vote", "vote_stake", "vote_free", "random", "confidence_rate", "confidence_gap")
    if rec.get("task") in SET_TASKS or sc.get("score") is None or sc.get("per_agent") is None:
        return {k: None for k in keys}
    cand = _candidates(rec)
    rate = (sum(1 for i in cand if stated(rec["finals"][i]["final"]) is not None) / len(cand)) if cand else None
    return {"top_confidence": top_confidence(rec), "vote": vote(rec), "vote_stake": vote_stake(rec),
            "vote_free": vote_free(rec), "random": random_pick(rec), "confidence_rate": rate,
            "confidence_gap": confidence_gap(rec)}


def _candidates_any(rec: dict) -> list[int]:
    return [i for i, f in enumerate(rec.get("finals") or []) if not f.get("failing") and f.get("final")]


def committed_key(rec: dict) -> tuple | None:
    """(normalised answer, backers, summed confidence) of the committed answer; ties go to the smallest answer."""
    clusters: dict = defaultdict(list)
    for i in _candidates_any(rec):
        k = _vote_key(rec, i)
        if k is not None:
            clusters[k].append(i)
    if not clusters:
        return None

    def conf(i):
        return stake(rec["finals"][i].get("final"))

    best = min(clusters, key=lambda k: (-len(clusters[k]), -sum(conf(i) for i in clusters[k]), k))
    return best, len(clusters[best]), sum(conf(i) for i in clusters[best])


def self_label(recs: list[dict], mode: str) -> str | None:
    """A label for one question from its runs without the reference answer ('commit' or 'majority')."""
    votes, total = Counter(), Counter()
    if mode == "commit":
        for r in recs:
            c = committed_key(r)
            if c is not None:
                votes[c[0]] += 1
                total[c[0]] += c[2]
    elif mode == "majority":
        for r in recs:
            for i in _candidates_any(r):
                k = _vote_key(r, i)
                if k is not None:
                    votes[k] += 1
    else:
        raise ValueError(mode)
    if not votes:
        return None
    return min(votes, key=lambda k: (-votes[k], -total[k], k))


def label_agreement(recs: list[dict], label: str | None, mode: str) -> float | None:
    """Share of the runs (or member answers) that back the label."""
    if label is None:
        return None
    if mode in ("commit", "review"):
        keys = [c[0] for c in (committed_key(r) for r in recs) if c is not None]
    elif mode == "majority":
        keys = [k for r in recs for k in (_vote_key(r, i) for i in _candidates_any(r)) if k is not None]
    else:
        raise ValueError(mode)
    return sum(k == label for k in keys) / len(keys) if keys else None


def relabel(rec: dict, label: str | None) -> dict:
    """The record rescored against a label; the reference score is kept under true_score."""
    sc = dict(rec.get("score") or {})
    per = []
    for i, f in enumerate(rec.get("finals") or []):
        if f.get("failing"):
            per.append(None)
        elif not f.get("final"):
            per.append(0.0)
        else:
            per.append(1.0 if label is not None and _vote_key(rec, i) == label else 0.0)
    valid = [p for p in per if p is not None]
    return {**rec, "true_score": rec.get("score"), "score": {**sc, "score": max(valid) if valid else 0.0,
                                                            "per_agent": per, "label": label}}
