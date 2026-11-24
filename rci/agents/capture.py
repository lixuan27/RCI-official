"""Captured token chains of a group run, stored next to its record for building training rows."""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

SUFFIX = ".chains.jsonl.gz"


def chains_path(record_path: Path) -> Path:
    record_path = Path(record_path)
    name = record_path.name
    if name.endswith(".json.gz"):
        name = name[:-len(".json.gz")]
    return record_path.with_name(name + SUFFIX)


def agent_chains(agent, l_max: int = 49152) -> list[dict]:
    """The agent's turns as merged sequences: token ids and the [start, length] span of each generation."""
    from rci.train.rollout import RolloutConfig, _samples_from_agent

    cfg = RolloutConfig(merge_prefix=True, include_summaries=False, l_max=l_max)
    out = []
    for s in _samples_from_agent(agent, 0.0, "", 0, cfg, random.Random(0)):
        out.append({"agent": agent.id, "ids": s.prompt_ids + s.completion_ids,
                    "spans": s.spans or [[len(s.prompt_ids), len(s.completion_ids)]]})
    return out


def _call_row(c) -> dict | None:
    if not c.token_ids or c.prompt_token_ids is None:
        return None
    return {"ids": list(c.prompt_token_ids) + list(c.token_ids), "spans": [[len(c.prompt_token_ids), len(c.token_ids)]]}


def write_chains(record_path: Path, agents, l_max: int = 49152, extra: list | None = None) -> None:
    """Write every member's chains (and `extra` calls given as (member, sample, call)) before the record is written."""
    path = chains_path(record_path)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt") as f:
        for a in agents:
            for c in agent_chains(a, l_max):
                f.write(json.dumps(c) + "\n")
        for member, j, c in extra or []:
            row = _call_row(c)
            if row:
                f.write(json.dumps({"extra": member, "sample": j, **row}) + "\n")
    tmp.replace(path)


def read_all(record_path: Path) -> list[dict]:
    with gzip.open(chains_path(record_path), "rt") as f:
        return [json.loads(line) for line in f]


def read_chains(record_path: Path) -> dict[int, list[dict]]:
    """Member id -> its chains ({"agent", "ids", "spans"})."""
    by: dict[int, list[dict]] = {}
    for c in read_all(record_path):
        if "agent" in c:
            by.setdefault(c["agent"], []).append(c)
    return by
