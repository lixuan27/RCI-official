"""Trainer configuration: a YAML file plus `--set key=value` overrides (values parsed as YAML)."""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field

import yaml

from rci.paths import models_dir, results_dir


@dataclass
class TrainConfig:
    """Every trainer option; YAML keys and `--set` overrides use these names."""
    run_id: str = "dev"
    out_root: str = ""
    model_path: str = "${RCI_MODELS}/Qwen3.5-4B"
    seed: int = 0

    task: str = "audit"
    task_kwargs: dict = field(default_factory=lambda: {"level": 2})
    source: dict = field(default_factory=lambda: {"kind": "auto", "min_contradictions": 4, "projects": None})
    mix: list = field(default_factory=list)
    primary_weight: int = 1
    primary_beta: float | None = None
    protocol: str = "blackboard"
    protocol_kwargs: dict = field(default_factory=dict)
    n_agents: int = 4
    trainees: str = "one"
    credit: str = "team"
    adv_mode: str = "grpo"
    commit: str = "top_confidence"
    silent_floor: bool = False
    beta: float = 0.5
    lam: float = 0.25
    k: int = 8
    questions_per_step: int = 8
    max_actions: int = 60
    max_steps: int = 60
    gen_limit: int = 8192
    enable_thinking: bool = True
    summarize: bool | None = None
    max_samples_per_agent: int | None = None
    include_summaries: bool = True
    merge_prefix: bool = False
    async_rollout: bool = False
    rollout_ahead: int = 1
    val_every: int = 0
    val_k: int = 4
    val_projects: list | None = None

    policy_urls: list = field(default_factory=list)
    peer_urls: list = field(default_factory=list)
    served_policy: str = "policy"
    served_peer: str = "peer"
    llm_concurrency: int = 256
    rollout_workers: int = 8

    lr: float = 1e-6
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eps_lo: float = 0.2
    eps_hi: float = 0.28
    tis_c: float = 2.0
    minibatches: int = 1
    ppo_prepass: bool = True
    l_max: int = 32768
    pad_multiple: int = 1024
    lm_chunk: int = 2048
    ckpt_min_len: int = 0
    total_steps: int = 200

    ckpt_every: int = 10
    hf_every: int = 0
    stop_check_every: int = 2
    keep_ckpts: int = 2
    wt_master_address: str = "127.0.0.1"
    wt_master_port: int = 31000
    wt_buffer_bytes: int = 1 << 30
    verify_sync_every: int = 0
    wandb_project: str = "rci"
    wandb_mode: str = "disabled"

    def validate(self) -> None:
        assert self.trainees in ("one", "all"), self.trainees
        assert self.credit in CREDITS, self.credit
        assert self.adv_mode in ("grpo", "raft"), self.adv_mode
        assert self.commit in ("top_confidence", "vote_stake", "vote_free"), self.commit
        assert self.rollout_ahead >= 1 and (self.rollout_ahead == 1 or self.async_rollout), \
            "rollout_ahead > 1 needs async_rollout"
        for m in self.mix:
            assert m.get("task") and int(m.get("weight", 1)) >= 1, m
        assert self.primary_weight >= 0 and (self.primary_weight or self.mix)
        assert self.minibatches >= 1, self.minibatches
        assert self.policy_urls, "policy_urls is empty"
        assert self.k >= 2 and self.questions_per_step >= 1
        assert self.l_max % self.pad_multiple == 0


CREDITS = ("team", "shapley", "shapley_flow")


def _expand(path: str) -> str:
    if "${RCI_MODELS}" in path:
        path = path.replace("${RCI_MODELS}", str(models_dir()))
    return os.path.expandvars(path)


def _parse_value(s: str):
    return yaml.safe_load(s)


def _set(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | None, overrides: list[str] | None = None) -> TrainConfig:
    raw: dict = {}
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set(raw, key.strip(), _parse_value(val))
    names = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    base = TrainConfig()
    if raw.get("task", base.task) != "audit":
        base.task_kwargs = {}
    for k, v in raw.items():
        cur = getattr(base, k)
        if isinstance(cur, dict) and isinstance(v, dict):
            v = {**cur, **v}
        elif isinstance(cur, float) and isinstance(v, (str, int)) and not isinstance(v, bool):
            v = float(v)
        elif isinstance(cur, int) and not isinstance(cur, bool) and isinstance(v, str):
            v = int(v)
        setattr(base, k, v)
    if isinstance(base.betas, list):
        base.betas = tuple(base.betas)
    base.model_path = _expand(base.model_path)
    base.out_root = _expand(base.out_root) if base.out_root else str(results_dir("train"))
    return base
