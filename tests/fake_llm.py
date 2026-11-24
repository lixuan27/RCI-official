from __future__ import annotations

import json
import random

from rci.agents.llm import ContextTooLong, LLMCall


class ScriptedLLM:
    def __init__(self, policy, seed: int = 0):
        self.policy = policy
        self.rng = random.Random(seed)
        self.n_calls = 0
        self.log: list[dict] = []

    async def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=16384, temperature=1.0,
                   top_p=None, top_k=None, seed=None, presence_penalty=None, enable_thinking=True, capture=False,
                   route_key=None, tag=None, alternatives=0, **kw) -> LLMCall:
        self.n_calls += 1
        names = [t["function"]["name"] for t in (tools or [])]
        self.log.append({"tag": tag, "n_messages": len(messages), "tools": names, "route_key": route_key,
                         "max_tokens": max_tokens,
                         "system": messages[0]["content"] if messages and messages[0]["role"] == "system" else None})
        if tag == "summary":
            return LLMCall(content="SUMMARY: earlier work condensed.", prompt_tokens=100, completion_tokens=8, tag=tag)
        out = self.policy(messages, names, route_key, tag)
        if isinstance(out, str):
            return LLMCall(content=out, prompt_tokens=sum(len(str(m.get("content", ""))) for m in messages) // 4,
                           completion_tokens=5, tag=tag)
        calls = [{"id": f"call_{self.n_calls}_{i}", "name": n, "arguments": json.dumps(a)} for i, (n, a) in enumerate(out)]
        call = LLMCall(content="", tool_calls=calls, reasoning="thinking...",
                       prompt_tokens=sum(len(str(m.get("content", ""))) for m in messages) // 4,
                       completion_tokens=20 * len(calls), tag=tag)
        if capture:
            call.prompt_token_ids, call.token_ids, call.logprobs = [1, 2, 3], [4, 5], [-0.1, -0.2]
        return call


class FakeChatLLM:
    def __init__(self, policy, tokenizer=None, max_model_len: int | None = None, seed: int = 0, raise_on=None):
        self.policy = policy
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len
        self.rng = random.Random(seed)
        self.raise_on = set(raise_on or ())
        self.n_calls = 0
        self.log: list[dict] = []

    def _call(self, out, tag) -> LLMCall:
        if isinstance(out, LLMCall):
            return out
        if isinstance(out, str):
            return LLMCall(content=out, tag=tag)
        if isinstance(out, dict):
            content, reasoning, pairs = out.get("content") or "", out.get("reasoning"), out.get("calls") or []
        else:
            content, reasoning, pairs = "", "thinking...", list(out)
        calls = [{"id": f"call_{self.n_calls}_{i}", "name": n,
                  "arguments": a if isinstance(a, str) else json.dumps(a)} for i, (n, a) in enumerate(pairs)]
        return LLMCall(content=content, reasoning=reasoning, tool_calls=calls, tag=tag)

    async def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=16384, temperature=1.0, top_p=None,
                   top_k=None, seed=None, presence_penalty=None, enable_thinking=True, capture=False, route_key=None,
                   tag=None, alternatives=0, want_token_ids=False, **kw) -> LLMCall:
        idx = self.n_calls
        self.n_calls += 1
        names = [t["function"]["name"] for t in (tools or [])]
        entry = {"index": idx, "tag": tag, "messages": messages, "tools": names, "route_key": route_key,
                 "max_tokens": max_tokens, "enable_thinking": enable_thinking, "raised": False}
        self.log.append(entry)
        ids = None
        if self.tokenizer is not None:
            ids = list(self.tokenizer.render(messages, tools=tools, enable_thinking=enable_thinking))
            entry["rendered_tokens"] = len(ids)
        over = (self.max_model_len is not None and ids is not None and len(ids) + max_tokens > self.max_model_len)
        if idx in self.raise_on or over:
            entry["raised"] = True
            raise ContextTooLong(f"This model's maximum context length is {self.max_model_len} tokens (fake)")
        if tag == "summary":
            call = LLMCall(content="SUMMARY: earlier work condensed.", prompt_tokens=100, completion_tokens=8, tag=tag)
        else:
            call = self._call(self.policy(messages, names, route_key, tag), tag)
            call.tag = call.tag or tag
        if self.tokenizer is not None:
            if not call.prompt_tokens:
                call.prompt_tokens = len(ids)
            if not call.completion_tokens:
                call.completion_tokens = len(self.tokenizer.output_ids(call))
            if want_token_ids or capture:
                call.prompt_token_ids = list(ids)
                call.token_ids = list(self.tokenizer.output_ids(call))
        else:
            if not call.prompt_tokens:
                call.prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
            if not call.completion_tokens:
                call.completion_tokens = 20 * len(call.tool_calls) if call.tool_calls else 5
            if capture:
                call.prompt_token_ids, call.token_ids, call.logprobs = [1, 2, 3], [4, 5], [-0.1, -0.2]
        entry["call"] = call
        return call
