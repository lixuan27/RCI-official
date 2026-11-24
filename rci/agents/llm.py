"""Async client for OpenAI-compatible (vLLM) chat servers with token accounting."""
from __future__ import annotations

import asyncio
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


class ContextTooLong(Exception):
    pass


class Router:
    """Sticky, load-aware assignment of route keys to replicas."""
    LIVE_S = 300.0
    MAX_KEYS = 50_000
    COOLDOWN_S = 30.0

    def __init__(self, n: int, clock=time.time):
        self.n, self.clock = n, clock
        self.keys: OrderedDict[str, list] = OrderedDict()
        self.inflight = [0] * n
        self.down_until = [0.0] * n

    def fail(self, i: int) -> None:
        self.down_until[i] = self.clock() + self.COOLDOWN_S

    def rebind(self, route_key: str | None, i: int) -> None:
        if route_key is not None and route_key in self.keys:
            self.keys[route_key][0] = i

    def pick(self, route_key: str | None) -> int:
        if self.n == 1:
            return 0
        now = self.clock()
        if route_key is not None and route_key in self.keys:
            r = self.keys[route_key]
            r[1] = now
            self.keys.move_to_end(route_key)
            return r[0]
        live = [0] * self.n
        for i, seen in self.keys.values():
            if now - seen < self.LIVE_S:
                live[i] += 1
        up = [i for i in range(self.n) if self.down_until[i] <= now] or list(range(self.n))
        best = min(up, key=lambda i: (live[i] + self.inflight[i], random.random()))
        if route_key is not None:
            self.keys[route_key] = [best, now]
            while len(self.keys) > self.MAX_KEYS:
                self.keys.popitem(last=False)
        return best


@dataclass
class LLMCall:
    content: str = ""
    reasoning: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    latency: float = 0.0
    prompt_token_ids: list[int] | None = None
    token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    error: str | None = None
    server: str | None = None
    tag: str | None = None
    top_logprobs: dict[str, float] | None = None
    new_prompt_tokens: int | None = None
    rendered_prompt_tokens: int | None = None
    out_ids: str | None = None

    TOKEN_FIELDS = ("prompt_token_ids", "token_ids", "logprobs")
    METER_FIELDS = ("new_prompt_tokens", "rendered_prompt_tokens", "out_ids")

    def summary(self) -> dict:
        out = {"tag": self.tag, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
               "finish_reason": self.finish_reason, "latency": round(self.latency, 3),
               "n_tool_calls": len(self.tool_calls), "error": self.error}
        for k in self.METER_FIELDS:
            if getattr(self, k) is not None:
                out[k] = getattr(self, k)
        return out

    def to_dict(self, tokens: bool = True) -> dict:
        """Every field as plain JSON; `tokens=False` leaves out the token ids and their log-probabilities."""
        from dataclasses import asdict
        d = asdict(self)
        if not tokens:
            for k in self.TOKEN_FIELDS:
                d.pop(k, None)
        for k in self.METER_FIELDS:
            if d.get(k) is None:
                d.pop(k, None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LLMCall":
        """The call a to_dict wrote (unknown keys ignored, missing ones defaulted; tool calls copied)."""
        from dataclasses import fields
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in (d or {}).items() if k in names}
        kw["tool_calls"] = [dict(tc) for tc in kw.get("tool_calls") or []]
        return cls(**kw)


def _get(obj: Any, name: str, default=None):
    v = getattr(obj, name, None)
    if v is None and getattr(obj, "model_extra", None):
        v = obj.model_extra.get(name)
    return default if v is None else v


class LLMClient:
    def __init__(self, base_urls: list[str] | str, model: str, *, api_key: str = "EMPTY",
                 timeout: float = 3600.0, max_concurrency: int = 256, max_retries: int = 30,
                 default_extra: dict | None = None, reasoning_effort: str | None = None, family: str | None = None):
        from openai import AsyncOpenAI

        from rci.agents.families import EFFORTS, FAMILIES, family_of

        self.base_urls = [base_urls] if isinstance(base_urls, str) else list(base_urls)
        self.model = model
        self.family = FAMILIES[family] if family is not None else family_of(model)
        if reasoning_effort is not None:
            if self.family.thinking != "effort":
                raise ValueError(f"reasoning_effort applies to effort families (gpt-oss); {model!r} is {self.family.name}")
            if reasoning_effort not in EFFORTS:
                raise ValueError(f"reasoning_effort {reasoning_effort!r} not in {EFFORTS}")
        self.reasoning_effort = reasoning_effort
        self.clients = [AsyncOpenAI(base_url=u, api_key=api_key, timeout=timeout, max_retries=0)
                        for u in self.base_urls]
        self.sem = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.default_extra = default_extra or {}
        self.router = Router(len(self.clients))

    def _pick(self, route_key: str | None) -> int:
        return self.router.pick(route_key)

    async def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
                   tool_choice: Any = "auto", max_tokens: int = 16384, temperature: float = 1.0,
                   top_p: float | None = None, top_k: int | None = None, seed: int | None = None,
                   presence_penalty: float | None = None, enable_thinking: bool | None = True,
                   capture: bool = False, route_key: str | None = None, tag: str | None = None, alternatives: int = 0,
                   want_token_ids: bool = False) -> LLMCall:
        """One chat completion."""
        import openai

        extra: dict = dict(self.default_extra)
        fam = self.family
        if fam.thinking == "template":
            if enable_thinking is not None:
                extra.setdefault("chat_template_kwargs", {})
                extra["chat_template_kwargs"] = {**extra["chat_template_kwargs"], fam.thinking_key: enable_thinking}
        else:
            effort = fam.effort(enable_thinking, self.reasoning_effort)
            if effort is not None:
                extra["reasoning_effort"] = effort
            if enable_thinking is False and fam.no_think_min_tokens:
                max_tokens = max(max_tokens, fam.no_think_min_tokens)
        if capture or want_token_ids:
            extra["return_token_ids"] = True
        if top_k is not None:
            extra["top_k"] = top_k
        kwargs: dict = dict(model=self.model, messages=messages, max_tokens=max_tokens,
                            temperature=temperature, extra_body=extra)
        if top_p is not None:
            kwargs["top_p"] = top_p
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if seed is not None:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if capture or alternatives:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = alternatives

        idx = self._pick(route_key)
        delay = 2.0
        last_err = None
        for attempt in range(self.max_retries):
            at = (idx + attempt) % len(self.clients)
            client = self.clients[at]
            t0 = time.time()
            try:
                async with self.sem:
                    self.router.inflight[at] += 1
                    try:
                        resp = await client.chat.completions.create(**kwargs)
                    finally:
                        self.router.inflight[at] -= 1
                call = self._parse(resp, time.time() - t0, self.base_urls[at], capture, tag, want_token_ids=want_token_ids)
                if at != idx:
                    self.router.rebind(route_key, at)
                if alternatives:
                    lp = getattr(resp.choices[0], "logprobs", None)
                    first = (getattr(lp, "content", None) or [None])[0]
                    call.top_logprobs = {t.token: float(t.logprob) for t in (first.top_logprobs if first else [])} or None
                return call
            except openai.BadRequestError as e:
                msg = str(e)
                if "maximum context length" in msg or "context length" in msg or "too long" in msg:
                    raise ContextTooLong(msg) from e
                last_err = msg
                if attempt >= 2:
                    break
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
                    openai.RateLimitError, openai.APIStatusError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):
                    self.router.fail(at)
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 1.7, 60.0)
        return LLMCall(error=f"LLM call failed after retries: {last_err}", tag=tag)

    @staticmethod
    def _parse(resp: Any, latency: float, server: str, capture: bool, tag: str | None, *,
               want_token_ids: bool = False) -> LLMCall:
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = []
        for tc in (msg.tool_calls or []):
            fn = tc.function
            tool_calls.append({"id": tc.id, "name": fn.name, "arguments": fn.arguments or "{}"})
        reasoning = _get(msg, "reasoning_content") or _get(msg, "reasoning")
        usage = resp.usage
        call = LLMCall(content=msg.content or "", reasoning=reasoning, tool_calls=tool_calls,
                       prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                       completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                       finish_reason=choice.finish_reason, latency=latency, server=server, tag=tag)
        if capture or want_token_ids:
            call.prompt_token_ids = _get(resp, "prompt_token_ids") or _get(choice, "prompt_token_ids")
            call.token_ids = _get(choice, "token_ids")
        if capture:
            lp = getattr(choice, "logprobs", None)
            content = getattr(lp, "content", None) if lp is not None else None
            if content:
                call.logprobs = [float(t.logprob) for t in content]
        return call


def lcp(a, b) -> int:
    """Length of the longest common prefix of two token-id sequences."""
    n = 0
    for x, y in zip(a or (), b or ()):
        if x != y:
            break
        n += 1
    return n


class PromptMeter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prev: dict = {}

    def new_tokens(self, route_key, ids: list[int]) -> int:
        """The new tokens of an already rendered prompt."""
        if route_key not in self.prev:
            return len(ids)
        return len(ids) - lcp(ids, self.prev[route_key])

    def measure(self, route_key, messages: list[dict], tools, enable_thinking) -> tuple[list[int], int]:
        """Render the prompt and return (ids, new)."""
        ids = list(self.tokenizer.render(messages, tools=tools, enable_thinking=enable_thinking))
        return ids, self.new_tokens(route_key, ids)

    def commit(self, route_key, ids: list[int], call) -> str | None:
        """Remember prompt + output of a sent call for the next LCP."""
        from rci.agents.tokenizer import out_ids_source
        if getattr(call, "error", None):
            out, src = [], None
        elif getattr(call, "token_ids", None):
            out, src = list(call.token_ids), "server"
        else:
            out, src = list(self.tokenizer.output_ids(call)), out_ids_source(self.tokenizer)
        self.prev[route_key] = list(ids) + out
        return src
