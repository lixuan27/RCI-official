"""Chat tokenizers for token caps measured on rendered chat text (a model's tokenizer or a stub)."""
from __future__ import annotations

import json
import os
import re
import zlib

_PIECE = re.compile(r"\w+|[^\w\s]")


def _content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type", "text") == "text" and isinstance(p.get("text"), str))
    return str(content)


def _call_parts(tc: dict) -> tuple[str, str]:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
    args = fn.get("arguments", "")
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return str(fn.get("name", "")), args


def server_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m.get("role") != "assistant" or "tool_calls" not in m or not isinstance(m.get("tool_calls"), list):
            out.append(m)
            continue
        m2 = dict(m)
        if not m2["tool_calls"]:
            m2.pop("tool_calls")
            out.append(m2)
            continue
        calls = []
        for item in m2["tool_calls"]:
            if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
                calls.append(item)
                continue
            fn = dict(item["function"])
            content = fn.get("arguments")
            if not content:
                parsed = {}
            elif isinstance(content, dict):
                parsed = content
            else:
                try:
                    parsed = json.loads(content) if isinstance(content, str) else content
                except ValueError:
                    parsed = {}
                if not isinstance(parsed, dict):
                    parsed = {}
            fn["arguments"] = parsed
            calls.append({**item, "function": fn})
        m2["tool_calls"] = calls
        out.append(m2)
    return out


class StubChatTokenizer:
    name = "stub"

    def encode(self, text: str) -> list[int]:
        return [zlib.crc32(p.encode("utf-8")) % 100000 + 16 for p in _PIECE.findall(text or "")]

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def truncate(self, text: str, n: int) -> str:
        """The longest verbatim prefix with at most n pieces: the text cut right after the n-th piece match."""
        text = text or ""
        if n <= 0:
            return ""
        for k, m in enumerate(_PIECE.finditer(text), 1):
            if k == n:
                return text[: m.end()]
        return text

    def _assistant_body(self, reasoning, content, tool_calls, enable_thinking=True) -> str:
        s = ""
        if reasoning and enable_thinking is not False:
            s += "<think>" + reasoning + "</think>\n"
        s += _content_text(content)
        for tc in tool_calls or []:
            name, args = _call_parts(tc)
            s += "<tool_call>" + name + " " + args + "</tool_call>"
        return s + "\n"

    def render_text(self, messages: list[dict], tools=None, enable_thinking=True, add_generation_prompt=True) -> str:
        """The rendered prompt as one string (render() encodes it)."""
        s = ""
        if tools:
            s += "<|tools|>" + json.dumps(tools, sort_keys=True, ensure_ascii=False) + "\n"
        for m in messages:
            role = m.get("role", "")
            s += "<|" + role + "|>\n"
            if role == "assistant":
                s += self._assistant_body(m.get("reasoning_content"), m.get("content"), m.get("tool_calls"),
                                          enable_thinking)
            else:
                s += _content_text(m.get("content"))
                for tc in m.get("tool_calls") or []:
                    name, args = _call_parts(tc)
                    s += "<tool_call>" + name + " " + args + "</tool_call>"
                s += "\n"
        if add_generation_prompt:
            s += "<|assistant|>\n"
        return s

    def render(self, messages: list[dict], tools=None, enable_thinking=True, add_generation_prompt=True) -> list[int]:
        return self.encode(self.render_text(messages, tools, enable_thinking, add_generation_prompt))

    def output_ids(self, call) -> list[int]:
        """The assistant rendering of a call's output without the role header."""
        return self.encode(self._assistant_body(getattr(call, "reasoning", None), getattr(call, "content", "") or "",
                                                getattr(call, "tool_calls", None) or []))


class HFChatTokenizer:
    def __init__(self, path: str):
        from transformers import AutoTokenizer
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        self.tok = AutoTokenizer.from_pretrained(path)

    def encode(self, text: str) -> list[int]:
        return list(self.tok(text or "", add_special_tokens=False)["input_ids"])

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def truncate(self, text: str, n: int) -> str:
        text = text or ""
        if n <= 0:
            return ""
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        if len(ids) <= n:
            return text
        k = n
        while k > 0:
            cut = text[: offsets[k - 1][1]]
            if self.count(cut) <= n:
                return cut
            k -= 1
        return ""

    def render(self, messages: list[dict], tools=None, enable_thinking=True, add_generation_prompt=True) -> list[int]:
        kw = dict(tools=tools or None, add_generation_prompt=add_generation_prompt, tokenize=True)
        if enable_thinking is not None:
            kw["enable_thinking"] = enable_thinking
        out = self.tok.apply_chat_template(server_messages(messages), **kw)
        if isinstance(out, dict) or hasattr(out, "keys"):
            out = out["input_ids"]
        return list(out)

    def output_ids(self, call) -> list[int]:
        """Approximate output ids (used only when the server returned none; the call is then tagged "approx")."""
        args = "".join(_call_parts(tc)[1] for tc in (getattr(call, "tool_calls", None) or []))
        return self.encode((getattr(call, "reasoning", None) or "") + (getattr(call, "content", "") or "") + args)


_CACHE: dict = {}


def load_tokenizer(spec):
    if isinstance(spec, str):
        if spec == "stub":
            return StubChatTokenizer()
        if spec not in _CACHE:
            _CACHE[spec] = HFChatTokenizer(spec)
        return _CACHE[spec]
    return spec


def out_ids_source(tokenizer) -> str:
    """The `out_ids` tag of a tokenizer fallback: "stub" for the stub, "approx" for anything else."""
    return "stub" if getattr(tokenizer, "name", None) == "stub" else "approx"
