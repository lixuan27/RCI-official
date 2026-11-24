"""Collect training samples from live group runs."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

from rci.agents.runner import GroupRun, RunConfig
from rci.ladder.commit import metrics as commit_metrics
from rci.train.credit import agent_credits


@dataclass
class Sample:
    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]
    reward: float
    advantage: float = 0.0
    question: str = ""
    rollout: int = 0
    agent: int = 0
    tag: str = "turn"
    meta: dict = field(default_factory=dict)
    spans: list | None = None

    @property
    def n_generations(self) -> int:
        return len(self.spans) if self.spans else 1


@dataclass
class RolloutConfig:
    trainees: str = "one"
    credit: str = "team"
    k: int = 8
    beta: float = 0.5
    lam: float = 0.25
    max_samples_per_agent: int | None = None
    include_summaries: bool = True
    merge_prefix: bool = False
    l_max: int = 32768
    adv_mode: str = "grpo"
    commit: str = "top_confidence"
    silent_floor: bool = False
    seed: int = 0


def merge_chains(samples: list[Sample], l_max: int) -> list[Sample]:
    """Merge consecutive samples (in call order) whose prompt starts with the previous prompt + completion."""
    out: list[Sample] = []
    seq: list[int] = []
    for s in samples:
        cur = out[-1] if out else None
        n = len(s.prompt_ids)
        if (cur is not None and n >= len(seq) and n + len(s.completion_ids) <= l_max
                and s.prompt_ids[: len(seq)] == seq):
            spans = cur.spans or [[len(cur.prompt_ids), len(cur.completion_ids)]]
            cur.prompt_ids, cur.completion_ids = list(s.prompt_ids), list(s.completion_ids)
            cur.logprobs = cur.logprobs + list(s.logprobs)
            cur.spans = spans + [[n, len(s.completion_ids)]]
        else:
            out.append(Sample(list(s.prompt_ids), list(s.completion_ids), list(s.logprobs), s.reward,
                              question=s.question, rollout=s.rollout, agent=s.agent, tag=s.tag, meta=dict(s.meta)))
        seq = out[-1].prompt_ids + out[-1].completion_ids
    return out


def _samples_from_agent(agent, reward: float, qid: str, r: int, cfg: RolloutConfig, rng: random.Random) -> list[Sample]:
    out = []
    for c in agent.calls:
        if c.error or not c.token_ids or c.prompt_token_ids is None or c.logprobs is None:
            continue
        if c.tag == "summary" and not cfg.include_summaries:
            continue
        out.append(Sample(list(c.prompt_token_ids), list(c.token_ids), list(c.logprobs), reward,
                          question=qid, rollout=r, agent=agent.id, tag=c.tag or "turn"))
    if cfg.merge_prefix:
        out = merge_chains(out, cfg.l_max)
    if cfg.max_samples_per_agent and len(out) > cfg.max_samples_per_agent:
        m = cfg.max_samples_per_agent
        w = len(out) / m
        out = rng.sample(out, m)
        for s in out:
            s.meta["weight"] = w
    return out


async def rollout_question(task, example: dict, protocol_factory, run_cfg: RunConfig, policy_llm, peer_llm,
                           cfg: RolloutConfig, step: int) -> tuple:
    """K group runs of one question, turned into training samples with credited rewards and advantages."""
    rng = random.Random(hash((cfg.seed, step, example["id"])))
    n = run_cfg.n_agents

    async def play(r: int):
        rc = RunConfig(**{**run_cfg.__dict__, "seed": rng.randrange(1 << 30), "capture": False})
        if cfg.trainees == "one":
            pid = rng.randrange(n)
            llms = {pid: policy_llm, -1: peer_llm}
        else:
            pid = None
            llms = {-1: policy_llm}
        run = GroupRun(task, example, protocol_factory(), rc, llms, run_id=f"train/s{step}/{example['id']}/r{r}")
        run.capture_agents = {pid} if pid is not None else "all"
        rec = await run.run()
        if rec["score"].get("score") is None:
            raise RuntimeError("training tasks need online scoring")
        return run, rec, pid

    plays = await asyncio.gather(*(play(r) for r in range(cfg.k)))
    def finish(r: int, run, rec: dict, pid):
        credits = agent_credits(rec, example, cfg.credit, beta=cfg.beta, lam=cfg.lam, commit=cfg.commit,
                                silent_floor=cfg.silent_floor)
        samples = []
        for a in run.agents:
            if pid is not None and a.id != pid:
                continue
            samples += _samples_from_agent(a, credits.get(a.id, 0.0), example["id"], r, cfg, rng)
        per = [x for x in (rec["score"].get("per_agent") or []) if x is not None]
        summary = {"question": example["id"], "task": task.name, "rollout": r, "score": rec["score"]["score"],
                   "policy_agent": pid,
                   "per_agent_mean": sum(per) / len(per) if per else None,
                   "distinct_pairs": rec["score"].get("n_reported_pairs"),
                   "exact_pairs": rec["score"].get("n_exact_pairs"),
                   "precision": rec["score"].get("precision"),
                   "recall": rec["score"].get("recall"),
                   "penalty": rec["score"].get("penalty"),
                   **{k: v for k, v in commit_metrics(rec).items()
                      if k in ("top_confidence", "vote", "vote_stake", "vote_free", "confidence_gap")},
                   "members": len(rec["finals"]),
                   "credits": credits, "n_samples": len(samples), "compute": rec["compute"],
                   "n_messages": len((rec.get("method_state") or {}).get("messages", []))}
        return samples, summary

    results = [finish(r, *plays[r]) for r in range(cfg.k)]
    samples = [s for ss, _ in results for s in ss]
    if cfg.adv_mode == "raft":
        raft_group(samples)
    else:
        normalize_group(samples)
    return samples, [m for _, m in results]


def normalize_group(samples: list[Sample], eps: float = 1e-6) -> None:
    """Group-relative advantage over all generations of one question (no gradient through stats)."""
    if not samples:
        return
    cs = [s.n_generations * float(s.meta.get("weight", 1.0)) for s in samples]
    tot = sum(cs)
    mu = sum(c * s.reward for c, s in zip(cs, samples)) / tot
    sd = (sum(c * (s.reward - mu) ** 2 for c, s in zip(cs, samples)) / tot) ** 0.5
    for s in samples:
        s.advantage = (s.reward - mu) / (sd + eps) if sd > 0 else 0.0


def raft_group(samples: list[Sample]) -> None:
    """Imitate wins: advantage 1 for the samples of the best rollouts, 0 otherwise."""
    if not samples:
        return
    best = max(s.reward for s in samples)
    for s in samples:
        s.advantage = 1.0 if best > 0 and s.reward >= best else 0.0
