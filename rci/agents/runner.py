"""Runs one group of agents on one example under one protocol, on a global step clock."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import inspect
import random
import time
from dataclasses import asdict, dataclass, field

from rci.agents.agent import Agent, AgentConfig
from rci.agents.llm import LLMClient


@dataclass
class RunConfig:
    n_agents: int = 1
    max_actions: int = 300
    max_steps: int = 400
    gen_limit: int = 16384
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    enable_thinking: bool | None = True
    summarize: bool = True
    seed: int = 0
    dropout_p: float = 0.0
    n_failing: int = 0
    failing_position: str = "first"
    capture: bool = False
    save_transcripts: bool = False
    method_kwargs: dict = field(default_factory=dict)

    def agent_config(self) -> AgentConfig:
        return AgentConfig(max_actions=self.max_actions, gen_limit=self.gen_limit, temperature=self.temperature,
                           top_p=self.top_p, top_k=self.top_k, presence_penalty=self.presence_penalty,
                           enable_thinking=self.enable_thinking,
                           summarize=self.summarize, capture=self.capture)


TASK_AGENT_KEYS = ("compact_at", "status_line", "meter_prompts", "deliver_on_final")


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class GroupRun:
    """One group run: the protocol builds the members, the clock steps them, the task scores their answers."""

    def __init__(self, task, example: dict, method, cfg: RunConfig, llm: LLMClient | dict[int, LLMClient],
                 run_id: str | None = None):
        self.task, self.example, self.method, self.cfg = task, example, method, cfg
        self._llm = llm
        self.run_id = run_id or f"{task.name}/{method.name}/N{cfg.n_agents}/s{cfg.seed}/{example['id']}"
        self.rng = random.Random(stable_seed(self.run_id, cfg.seed))
        self.dropout_rng = random.Random(stable_seed(self.run_id, cfg.seed, "dropout"))
        self.agents: list[Agent] = []
        self.capture_agents: set[int] | str = set()
        self.step = 0
        self.state: dict = {}
        self.step_log: list[dict] = []
        self.extra_calls: list = []
        if getattr(task, "event_log", False):
            self.state["event_log"] = True
            self.state["event_seq"] = 0

    def llm_for(self, agent_id: int) -> LLMClient:
        if isinstance(self._llm, dict):
            return self._llm.get(agent_id, self._llm.get(-1))
        return self._llm

    def agent_config(self, **overrides) -> AgentConfig:
        """A member's config: the run's own, then the settings the task drives, then `overrides`."""
        cfg = self.cfg.agent_config()
        task_kw = {k: getattr(self.task, k) for k in TASK_AGENT_KEYS if getattr(self.task, k, None) is not None}
        if task_kw or overrides:
            cfg = dataclasses.replace(cfg, **{**task_kw, **overrides})
        return cfg

    def add_agent(self, system_prompt: str, task_prompt: str, tools, *, failing: bool = False,
                  role: str | None = None, cfg: AgentConfig | None = None) -> Agent:
        aid = len(self.agents)
        cfg = cfg or self.agent_config()
        if self.capture_agents == "all" or aid in self.capture_agents:
            cfg.capture = True
        agent = Agent(aid, system_prompt, task_prompt, tools, self.llm_for(aid), cfg, self, failing=failing, role=role)
        self.agents.append(agent)
        return agent

    async def run(self) -> dict:
        t0 = time.time()
        self.method.build(self)
        await self._steps(0)
        why = {"reason": "step_cap"} if self.state.get("event_log") is True else {}
        await asyncio.gather(*(a.force_final(self.step, **why) for a in self.agents
                               if not a.done and self.method.needs_final(self, a)))
        for a in self.agents:
            a.done = True
            self._notify_done(a)
        finals = self.method.collect(self)
        score = self.task.score_group(self.example, finals, self)
        if inspect.isawaitable(score):
            score = await score
        return self.record(score, finals, time.time() - t0)

    async def _steps(self, start: int) -> int:
        for step in range(start, start + self.cfg.max_steps):
            self.step = step
            active = [a for a in self.agents if not a.done]
            if not active or self.method.finished(self):
                return step
            await self.method.before_step(self, step)
            runnable = [a for a in active if not a.is_waiting(step)]
            present = [a for a in runnable
                       if self.cfg.dropout_p <= 0 or self.dropout_rng.random() >= self.cfg.dropout_p]
            plans = [(a, self.method.injection(self, a, step), self.method.live_block(self, a, step)) for a in present]
            await asyncio.gather(*(self._go(a, inj, live, step) for a, inj, live in plans))
            await self.method.after_step(self, step)
            self.step_log.append({"step": step, "active": len(active), "runnable": len(runnable),
                                  "present": len(present)})
        return start + self.cfg.max_steps

    async def _go(self, agent: Agent, injection, live, step: int) -> None:
        await agent.turn(step, injection, live)
        self.method.on_turn_end(self, agent, step)
        if agent.done:
            self._notify_done(agent)

    def _notify_done(self, agent: Agent) -> None:
        if not agent.done_notified:
            agent.done_notified = True
            self.method.on_agent_done(self, agent)

    def compute(self) -> dict:
        tot = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}
        new, metered = 0, False
        for c in [c for a in self.agents for c in a.calls] + self.extra_calls:
            tot["prompt_tokens"] += c.prompt_tokens
            tot["completion_tokens"] += c.completion_tokens
            tot["n_calls"] += 1 if c.error != "injected-failure" else 0
            if getattr(c, "new_prompt_tokens", None) is not None:
                new, metered = new + c.new_prompt_tokens, True
        tot["total_tokens"] = tot["prompt_tokens"] + tot["completion_tokens"]
        if metered:
            tot["new_prompt_tokens"] = new
        return tot

    def record(self, score: dict, finals: list[dict], wall: float) -> dict:
        rec = {
            "run_id": self.run_id, "task": self.task.name, "example_id": self.example["id"],
            "method": self.method.name, "config": asdict(self.cfg), "score": score, "finals": finals,
            "compute": self.compute(), "n_steps": self.step + 1, "wall_time": round(wall, 1),
            "agents": [a.record() for a in self.agents], "method_state": self.method.dump(self),
            "step_log": self.step_log,
        }
        if self.cfg.save_transcripts:
            rec["transcripts"] = {a.id: a.transcript() for a in self.agents}
        return rec
