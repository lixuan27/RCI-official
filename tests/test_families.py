import asyncio
import gzip
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rci.agents import families as F
from rci.agents.llm import LLMClient

ROOT = Path(__file__).resolve().parents[1]
QWEN_FLAGS = "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"


def old_payload(model, default_extra, messages, *, tools=None, tool_choice="auto", max_tokens=16384, temperature=1.0,
                top_p=None, top_k=None, seed=None, presence_penalty=None, enable_thinking=True, capture=False,
                alternatives=0, want_token_ids=False, **_):
    extra = dict(default_extra or {})
    if enable_thinking is not None:
        extra.setdefault("chat_template_kwargs", {})
        extra["chat_template_kwargs"] = {**extra["chat_template_kwargs"], "enable_thinking": enable_thinking}
    if capture or want_token_ids:
        extra["return_token_ids"] = True
    if top_k is not None:
        extra["top_k"] = top_k
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, extra_body=extra)
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
    return kwargs


def fake_openai(monkeypatch, created):
    mod = type(sys)("openai")

    class APIError(Exception):
        pass

    for n in ("BadRequestError", "APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError",
              "APIStatusError"):
        setattr(mod, n, type(n, (APIError,), {}))

    class Completions:
        async def create(self, **kw):
            created.append(kw)
            msg = SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None, model_extra=None)
            choice = SimpleNamespace(message=msg, finish_reason="stop", token_ids=[9, 8], logprobs=None, model_extra=None)
            return SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
                                   prompt_token_ids=[1, 2, 3], model_extra=None)

    class AsyncOpenAI:
        def __init__(self, **kw):
            self.chat = SimpleNamespace(completions=Completions())

    mod.AsyncOpenAI = AsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)


TOOLS = [{"type": "function", "function": {"name": "submit_answer", "parameters": {"type": "object", "properties": {}}}}]
NAMED = {"type": "function", "function": {"name": "submit_answer"}}
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]


def call_grid():
    for think, capture, top_k, top_p, pp, seed, tools, alt, wti, mt in itertools.product(
            (True, False, None), (False, True), (None, 20), (None, 0.95), (None, 1.5), (None, 7), (None, "auto", "named"),
            (0, 20), (False, True), (16384, 8)):
        kw = dict(enable_thinking=think, capture=capture, top_k=top_k, top_p=top_p, presence_penalty=pp, seed=seed,
                  alternatives=alt, want_token_ids=wti, max_tokens=mt, temperature=1.0, tag="turn", route_key="r/0")
        if tools:
            kw.update(tools=TOOLS, tool_choice=NAMED if tools == "named" else "auto")
        yield kw


def run_calls(monkeypatch, model, grid, default_extra=None, **client_kw):
    created = []
    fake_openai(monkeypatch, created)

    async def go():
        c = LLMClient("http://fake/v1", model, default_extra=default_extra, **client_kw)
        for kw in grid:
            await c.chat(MSGS, **kw)

    asyncio.run(go())
    return created


@pytest.mark.parametrize("model", ["qwen3.8-27b", "qwen3.5-9b", "some-finetune", "m", "fake"])
@pytest.mark.parametrize("default_extra", [None, {"chat_template_kwargs": {"x": 1}, "min_p": 0.0}])
def test_qwen_request_is_byte_identical_to_before(monkeypatch, model, default_extra):
    grid = list(call_grid())
    got = run_calls(monkeypatch, model, grid, default_extra=default_extra)
    want = [old_payload(model, default_extra, MSGS, **kw) for kw in grid]
    assert len(got) == len(want) == len(grid)
    for g, w in zip(got, want):
        assert json.dumps(g) == json.dumps(w)


def test_glm_thinking_goes_through_the_chat_template_as_for_qwen(monkeypatch):
    grid = list(call_grid())[:200]
    got = run_calls(monkeypatch, "glm-4.5-air-fp8", grid)
    assert [json.dumps(g) for g in got] == [json.dumps(old_payload("glm-4.5-air-fp8", None, MSGS, **kw)) for kw in grid]


def test_gptoss_sends_a_top_level_reasoning_effort_and_never_the_template_switch(monkeypatch):
    grid = [dict(enable_thinking=True), dict(enable_thinking=False, max_tokens=8), dict(enable_thinking=None),
            dict(enable_thinking=False, max_tokens=20000), dict(enable_thinking=True, max_tokens=8)]
    got = run_calls(monkeypatch, "gpt-oss-120b", grid)
    assert [g["extra_body"].get("reasoning_effort") for g in got] == ["medium", "low", None, "low", "medium"]
    assert all("chat_template_kwargs" not in g["extra_body"] for g in got)
    assert [g["max_tokens"] for g in got] == [16384, 4096, 16384, 20000, 8]
    hi = run_calls(monkeypatch, "gpt-oss-120b", grid, reasoning_effort="high")
    assert [g["extra_body"].get("reasoning_effort") for g in hi] == ["high", "low", "high", "low", "high"]
    for g, kw in zip(got, grid):
        w = old_payload("gpt-oss-120b", None, MSGS, **kw)
        w["extra_body"].pop("chat_template_kwargs", None)
        g = json.loads(json.dumps(g))
        g["extra_body"].pop("reasoning_effort", None)
        g.pop("max_tokens"), w.pop("max_tokens")
        assert g == w


def test_reasoning_effort_is_refused_where_it_means_nothing(monkeypatch):
    fake_openai(monkeypatch, [])
    with pytest.raises(ValueError):
        LLMClient("http://fake/v1", "qwen3.8-27b", reasoning_effort="high")
    with pytest.raises(ValueError):
        LLMClient("http://fake/v1", "glm-4.5-air-fp8", reasoning_effort="high")
    with pytest.raises(ValueError):
        LLMClient("http://fake/v1", "gpt-oss-120b", reasoning_effort="none")

    async def forced():
        return LLMClient("http://fake/v1", "m", family="gptoss").family.name

    assert asyncio.run(forced()) == "gptoss"


def test_family_lookup_by_name_and_prefix():
    assert [F.family_name(m) for m in ("qwen3.8-27b", "qwen3.5-4b", "some-finetune", "fake", "", None)] == ["qwen"] * 6
    assert F.family_name("gpt-oss-120b") == F.family_name("GPT-OSS-20B") == "gptoss"
    assert F.family_name("glm-4.5-air-fp8") == F.family_name("glm-4.6") == "glm45"
    q = F.FAMILIES["qwen"]
    assert " ".join(q.parser_flags) == QWEN_FLAGS and q.thinking == "template" and q.thinking_key == "enable_thinking"
    g = F.FAMILIES["gptoss"]
    assert g.parser_flags == ["--reasoning-parser", "openai_gptoss", "--enable-auto-tool-choice", "--tool-call-parser", "openai"]
    assert (g.effort(True), g.effort(False), g.effort(None), g.effort(True, "high"), g.effort(False, "high")) == (
        "medium", "low", None, "high", "low")
    assert g.task_effort == {"gpqa": "high"} and set(g.sampling) == {"temperature", "top_p"}
    z = F.FAMILIES["glm45"]
    assert z.parser_flags[1] == z.parser_flags[4] == "glm45" and not z.sampling_verified
    assert z.env == {"VLLM_ENFORCE_STRICT_TOOL_CALLING": "0"} and not z.named_tool_choice
    assert "VLLM_ENFORCE_STRICT_TOOL_CALLING" not in g.env and "VLLM_ENFORCE_STRICT_TOOL_CALLING" not in q.env
    assert F.record_info("qwen3.8-27b") is None and F.record_info("glm-4.5-air-fp8") == {"name": "glm45"}
    assert F.record_info("gpt-oss-120b", "high") == {"name": "gptoss", "reasoning_effort": "high", "effort_off": "low"}
    assert F.record_info("gpt-oss-120b", "high", thinking=False) == {"name": "gptoss", "reasoning_effort": "low",
                                                                     "effort_off": "low"}
    assert F.record_info("gpt-oss-120b", None, thinking=False)["reasoning_effort"] == "low"
    assert F.record_info("glm-4.5-air-fp8", None, thinking=False) == {"name": "glm45"}


def test_weight_size_comes_from_the_model_index(tmp_path):
    assert F.index_weight_bytes(None) is None and F.index_weight_bytes(str(tmp_path)) is None
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 65248815744},
                                                                       "weight_map": {}}))
    assert F.index_weight_bytes(str(tmp_path)) == 65248815744


OLD_ARGV = ["--enable-prefix-caching"] + QWEN_FLAGS.split() + ["--foo", "1"]


def put(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(rec, f)


def test_settings_mismatch_tells_families_apart_and_never_two_qwen_names(tmp_path):
    from rci.run import settings_mismatch
    p = tmp_path / "r.json.gz"
    put(p, {"config": {"method_kwargs": {}}, "task_kwargs": {}, "model": "qwen3.8-27b"})
    assert settings_mismatch(p, {}, {}) is None and settings_mismatch(p, {}, {}, model="qwen3.8-27b") is None
    assert settings_mismatch(p, {}, {}, model="some-alias") is None
    assert "family" in settings_mismatch(p, {}, {}, model="gpt-oss-120b")
    put(p, {"config": {"method_kwargs": {}}, "task_kwargs": {}, "model": "gpt-oss-120b",
            "family": {"name": "gptoss", "reasoning_effort": "medium", "effort_off": "low"}})
    assert settings_mismatch(p, {}, {}, model="gpt-oss-120b") is None
    assert "family" in settings_mismatch(p, {}, {}, model="gpt-oss-120b", reasoning_effort="high")
    assert settings_mismatch(p, {}, {}, model="gpt-oss-120b", reasoning_effort="medium") is None
    assert "family" in settings_mismatch(p, {}, {}, model="qwen3.8-27b")
    assert "family" in settings_mismatch(p, {}, {}, model="gpt-oss-120b", thinking=False)
    put(p, {"config": {"method_kwargs": {}}, "task_kwargs": {}, "model": "gpt-oss-120b",
            "family": {"name": "gptoss", "reasoning_effort": "low", "effort_off": "low"}})
    assert settings_mismatch(p, {}, {}, model="gpt-oss-120b", thinking=False) is None
    assert settings_mismatch(p, {}, {}, model="gpt-oss-120b", reasoning_effort="high", thinking=False) is None
    assert "family" in settings_mismatch(p, {}, {}, model="gpt-oss-120b")
    put(p, {"config": {"method_kwargs": {}}, "task_kwargs": {}, "model": "glm-4.5-air-fp8"})
    assert "family" in settings_mismatch(p, {}, {}, model="glm-4.5-air-fp8")
    put(p, {"config": {"method_kwargs": {}}, "task_kwargs": {}, "model": "a,b"})
    assert settings_mismatch(p, {}, {}, model="qwen3.8-27b") is None


def test_run_eval_family_sampling_presets_leave_the_generation_limit():
    from rci.run import SAMPLING
    assert SAMPLING["qwen_thinking"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                                         "gen_limit": 32768}
    assert SAMPLING["gptoss"] == {"temperature": 1.0, "top_p": 1.0} and "gen_limit" not in SAMPLING["glm45"]


def test_mad_strips_a_leaked_harmony_call_and_keeps_qwen_behaviour():
    from rci.protocols.debate import TOOL_CALL_TEXT
    q = 'I think B.\n<tool_call>\n<function=submit_answer>\n<parameter=answer>B</parameter>\n</function>\n</tool_call>'
    assert TOOL_CALL_TEXT.sub("", q).strip() == "I think B."
    assert TOOL_CALL_TEXT.sub("", "partial <tool_call> x").strip() == "partial"
    h = 'Answer B.<|start|>assistant<|channel|>commentary to=functions.submit_answer <|constrain|>json<|message|>{"answer":"B"}<|call|> after'
    assert TOOL_CALL_TEXT.sub("", h) == "Answer B. after"
    assert TOOL_CALL_TEXT.sub("", 'x <|channel|>commentary to=functions.f {"a":1}').strip() == "x"
    plain = "The channel to functions is commentary; no call here."
    assert TOOL_CALL_TEXT.sub("", plain) == plain
