"""Loading run records and pricing them for the equal-budget ladder."""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
from pathlib import Path

from rci.ladder.commit import vote_stake
from rci.ladder.curves import stakeless

COSTS = ("completion", "total", "agents")
_JUDGE_CACHES: dict = {}


def judged(rec: dict, root: str) -> dict | None:
    """A BrowseComp-Plus record still waiting for its judge, completed from the grader's cache, or None."""
    sc = rec.get("score") or {}
    if not sc.get("pending_judge"):
        return rec
    if rec.get("task") != "browsecomp":
        return None
    if root not in _JUDGE_CACHES:
        cp = os.path.join(root, "judge_cache.jsonl")
        cache = {}
        if os.path.exists(cp):
            with open(cp) as fh:
                for line in fh:
                    row = json.loads(line)
                    cache[row["key"]] = row["correct"]
        _JUDGE_CACHES[root] = cache
    from rci.tasks.browsecomp.grade import response_text

    per = []
    for fin in rec.get("finals") or []:
        if fin.get("failing"):
            per.append(None)
        elif not isinstance(fin.get("final"), dict) or not str(fin["final"].get("answer") or "").strip():
            per.append(0.0)
        else:
            key = hashlib.sha1(f"{rec['example_id']}\n{response_text(fin['final'])}".encode()).hexdigest()
            if key not in _JUDGE_CACHES[root]:
                return None
            per.append(1.0 if _JUDGE_CACHES[root][key] else 0.0)
    valid = [x for x in per if x is not None]
    return {**rec, "score": {"score": max(valid) if valid else 0.0, "per_agent": per}}


def load(root: str, protocol: str, n: int) -> dict[str, list[dict]]:
    """example id -> records of one cell, read from <root>/<protocol>/N<n>/s*/."""
    out: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(os.path.join(root, protocol, f"N{n}", "s*", "*.json*"))):
        if not f.endswith((".json", ".json.gz")) or f.endswith(".chains.jsonl.gz"):
            continue
        with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
            rec = judged(json.load(fh), root)
        if rec is None:
            continue
        key = os.path.basename(f).split(".json")[0]
        out.setdefault(key, []).append(rec)
    return out


def records(root: str, protocol: str, n: int) -> dict[str, list[dict]]:
    """load() restricted to graded records."""
    out = {}
    for k, recs in load(root, protocol, n).items():
        keep = [r for r in recs if (r.get("score") or {}).get("per_agent") and (r.get("score") or {}).get("score") is not None]
        if keep:
            out[k] = keep
    return out


def cost_of(rec: dict, kind: str) -> float:
    """A run's cost in completion tokens, total tokens, or members paid for ('agents')."""
    c = rec.get("compute") or {}
    if kind == "agents":
        return float((rec.get("config") or {}).get("n_agents") or len(rec.get("finals") or []))
    if kind == "completion":
        return float(c.get("completion_tokens", 0))
    if kind == "total":
        if c.get("total_tokens") is not None:
            return float(c["total_tokens"])
        return float(c.get("prompt_tokens", 0)) + float(c.get("completion_tokens", 0))
    raise ValueError(f"unknown cost {kind!r}")


def review_path(review: Path, rec: dict) -> Path:
    return review / f"s{(rec.get('config') or {}).get('seed', 0)}_{rec['example_id']}.json"


def answer_of(rec: dict, set_task: bool = False, review: Path | None = None, free: bool = False) -> float | None:
    """The score of the answer the group commits to, or of a reviewer's choice when `review` is given."""
    if set_task:
        return float(rec["score"]["score"])
    y = float(vote_stake(stakeless(rec) if free else rec))
    if review is None:
        return y
    from rci.ladder.reviewer import pick

    p = review_path(review, rec)
    return pick(json.loads(p.read_text()), y)[0] if p.exists() else None


def review_cost(rec: dict, review: Path | None, kind: str, scale: float) -> float:
    """What the reviewer spent on this run, in the members' unit."""
    if review is None:
        return 0.0
    p = review_path(review, rec)
    if not p.exists():
        return 0.0
    r = json.loads(p.read_text())
    if kind == "agents":
        return scale * (1.0 if r.get("reviewed") else 0.0)
    return scale * (r.get("completion_tokens", 0) + (r.get("prompt_tokens", 0) if kind == "total" else 0))
