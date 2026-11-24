"""LongBench v2: multiple-choice questions over long contexts that members read through tools."""
from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from pathlib import Path

from rci import paths
from rci.agents.tooling import STR, AgentCtx, ToolResult, ToolSpec
from rci.agents.tokens import count_tokens, truncate_to_tokens
from rci.tasks.base import Task
from rci.tasks.gpqa.task import norm_letter

LETTERS = "ABCD"

SECTION_CHARS, READ_TOKENS, TOP_K, SNIPPET = 6000, 2000, 5, 200
K1, B = 0.9, 0.4
SECTION_RE = re.compile(r"^Section(\d+)$")
_WORD = re.compile(r"\w+")
_STOP = frozenset("a an and are as at be but by for if in into is it no not of on or such that the their then "
                  "there these they this to was will with what which who how many much does do did".split())

SYSTEM = """You answer one multiple-choice question about a long document (a book, a report, a set of papers, a
dialogue history, a code repository, a table, ...). The document is far longer than you can read in full: it is
split into numbered sections, Section1 to SectionN, and you inspect it with tools.

Tools:
- search_document(query): keyword (BM25) search over the sections; returns the {top_k} best-matching sections with
  a short snippet each. Search for names, identifiers and distinctive terms from the question and the options.
- read_section(section): the text of one section (up to about {read_tokens} tokens).
Your earlier work is summarized from time to time, so write down what each section told you before moving on.

Exactly one option is correct. Find the passages that decide between the options; do not guess from the options'
wording alone. When you are confident, call `submit_answer` with the letter of the correct option (A, B, C or D)."""


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def split_sections(text: str, cap: int = SECTION_CHARS) -> list[str]:
    """Consecutive paragraphs packed up to `cap` characters; an over-long paragraph is cut at lines, then hard."""
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= cap:
            pieces.append(para)
            continue
        buf = ""
        for line in para.splitlines():
            while len(line) > cap:
                if buf:
                    pieces.append(buf)
                    buf = ""
                pieces.append(line[:cap])
                line = line[cap:]
            if len(buf) + len(line) + 1 > cap and buf:
                pieces.append(buf)
                buf = ""
            buf = f"{buf}\n{line}" if buf else line
        if buf:
            pieces.append(buf)
    out, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > cap:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out or [""]


class SectionDoc:
    def __init__(self, text: str):
        self.sections = split_sections(text)
        self.n = len(self.sections)
        self.n_words = len(text.split())
        self.tf = [Counter(tokenize(s)) for s in self.sections]
        self.dl = [sum(c.values()) for c in self.tf]
        self.avgdl = (sum(self.dl) / self.n) or 1.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {w: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def search(self, query: str, k: int = TOP_K) -> list[tuple[int, float]]:
        q = list(dict.fromkeys(tokenize(query)))
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (K1 + 1) / (f + K1 * (1 - B + B * self.dl[i] / self.avgdl))
            if s > 0:
                scores.append((i + 1, s))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:k]

    def snippet(self, sec: int, query: str, width: int = SNIPPET) -> str:
        text = re.sub(r"\s+", " ", self.sections[sec - 1])
        q = set(tokenize(query))
        low = text.lower()
        best, best_s = 0, -1
        for m in _WORD.finditer(low):
            if m.group() not in q:
                continue
            start = max(0, m.start() - width // 4)
            s = len(q.intersection(_WORD.findall(low[start:start + width])))
            if s > best_s:
                best, best_s = start, s
        out = text[best:best + width].strip()
        return ("..." if best > 0 else "") + out + ("..." if best + width < len(text) else "")


_LOCK = threading.Lock()
_DOCS: dict[tuple[str, str], SectionDoc] = {}


def shared_doc(root: str, doc_id: str) -> SectionDoc:
    key = (root, doc_id)
    with _LOCK:
        if key not in _DOCS:
            if len(_DOCS) >= 16:
                _DOCS.clear()
            _DOCS[key] = SectionDoc((Path(root) / "ctx" / f"{doc_id}.txt").read_text())
        return _DOCS[key]


def parse_section(v, n: int) -> tuple[int | None, str | None]:
    if isinstance(v, str):
        s = v.strip()
        m = SECTION_RE.match(s)
        s = m.group(1) if m else s
        if not s.isdigit():
            return None, f"Error: section must be a number such as 12 or Section12, got {v!r}."
        v = int(s)
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, bool) or not isinstance(v, int):
        return None, f"Error: section must be a number such as 12 or Section12, got {v!r}."
    if not 1 <= v <= n:
        return None, f"Error: Section{v} does not exist; this document has Section1 to Section{n}."
    return v, None


class Reader:
    def __init__(self, doc: SectionDoc, read_tokens: int):
        self.d, self.read_tokens = doc, read_tokens
        self.queries: list[str] = []
        self.read: list[int] = []

    def search(self, query: str) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult("Error: empty query.", ok=False)
        self.queries.append(query)
        hits = self.d.search(query)
        if not hits:
            return ToolResult(f"No section matches \"{query}\".")
        lines = [f"Top {len(hits)} sections for \"{query}\":"]
        lines += [f"Section{s} [score {sc:.2f}]: {self.d.snippet(s, query)}" for s, sc in hits]
        return ToolResult("\n".join(lines))

    def read_section(self, section) -> ToolResult:
        s, err = parse_section(section, self.d.n)
        if err:
            return ToolResult(err, ok=False)
        if s not in self.read:
            self.read.append(s)
        text = self.d.sections[s - 1]
        head = f"Section{s} of {self.d.n}:\n"
        if count_tokens(text) <= self.read_tokens:
            return ToolResult(head + text)
        return ToolResult(head + truncate_to_tokens(text, self.read_tokens) + "\n[section truncated]")


SYSTEM_FULL = """You answer one multiple-choice question about a document (a book, a report, a set of papers, a
dialogue history, a code repository, a table, ...). The whole document is given with the question: read it carefully.
It is also split into numbered sections, Section1 to SectionN, which you can search and re-read with tools:
- search_document(query): keyword (BM25) search over the sections; returns the {top_k} best-matching sections with
  a short snippet each.
- read_section(section): the text of one section (up to about {read_tokens} tokens).

Exactly one option is correct. Find the passages that decide between the options; do not guess from the options'
wording alone. When you are confident, call `submit_answer` with the letter of the correct option (A, B, C or D)."""
FULL_TOKENS = 65536
_FULL: dict = {}
_FULL_LOCK = threading.Lock()


def full_text(root: str, doc_id: str, n_chars: int | None = None, limit: int = FULL_TOKENS) -> str | None:
    """The document's text when it is at most `limit` o200k tokens, else None (cached per document and limit)."""
    key = (root, doc_id, limit)
    with _FULL_LOCK:
        if key in _FULL:
            return _FULL[key]
    if n_chars is not None and n_chars > 8 * limit:
        text = None
    else:
        text = (Path(root) / "ctx" / f"{doc_id}.txt").read_text()
        text = text if count_tokens(text) <= limit else None
    with _FULL_LOCK:
        _FULL[key] = text
    return text


class LongBenchV2(Task):
    name = "longbench2"
    msg_token_limit = 256
    max_actions = 100
    max_steps = 120
    summarize = True

    deliberate_gen_limit = 32768

    def __init__(self, root: str = "", length: str | None = None, difficulty: str | None = None,
                 read_tokens: int = READ_TOKENS, full_tokens: bool | int = True, file: str = "samples.jsonl"):
        assert length in (None, "short", "medium", "long"), length
        assert difficulty in (None, "easy", "hard"), difficulty
        self.root = root or str(paths.data_dir("longbench2"))
        self.length, self.difficulty, self.read_tokens = length, difficulty, read_tokens
        self.full_tokens = full_tokens
        self.file = file
        self._rows: list[dict] | None = None

    def examples(self, limit: int | None = None) -> list[dict]:
        if self._rows is None:
            p = Path(self.root) / self.file
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: run scripts/data/prepare_longbench2.py")
            with open(p) as f:
                self._rows = [json.loads(l) for l in f if l.strip()]
        rows = [r for r in sorted(self._rows, key=lambda r: r["idx"])
                if (self.length is None or r["length"] == self.length)
                and (self.difficulty is None or r["difficulty"] == self.difficulty)]
        rows = rows[:limit] if limit else rows
        return [{**r, "id": f"lb2_{r['_id']}"} for r in rows]

    def doc(self, example: dict) -> SectionDoc:
        return shared_doc(self.root, example.get("ctx_id") or example["_id"])

    def full(self, example: dict) -> str | None:
        if not self.full_tokens:
            return None
        limit = FULL_TOKENS if self.full_tokens is True else int(self.full_tokens)
        return full_text(self.root, example.get("ctx_id") or example["_id"], example.get("n_chars"), limit)

    def deliberates(self, example: dict) -> bool:
        return self.full(example) is not None

    def system_prompt(self, example: dict) -> str:
        sys_ = SYSTEM_FULL if self.full(example) is not None else SYSTEM
        return sys_.format(top_k=TOP_K, read_tokens=self.read_tokens)

    def task_prompt(self, example: dict, agent_id: int, n_agents: int, rng) -> str:
        d = self.doc(example)
        opts = "\n".join(f"{L}) {example['choice_' + L]}" for L in LETTERS)
        full = self.full(example)
        if full is not None:
            return (f"Document: {example['domain']} / {example['sub_domain']}; about {d.n_words:,} words, given in full "
                    f"below and also as {d.n} sections (Section1 to Section{d.n}).\n\n<document>\n{full}\n</document>"
                    f"\n\nQuestion: {example['question']}\n\n{opts}")
        return (f"Document: {example['domain']} / {example['sub_domain']}; about {d.n_words:,} words in {d.n} sections "
                f"(Section1 to Section{d.n}).\n\nQuestion: {example['question']}\n\n{opts}")

    def tools(self, example: dict, run) -> list[ToolSpec]:
        doc = self.doc(example)

        def rd(ctx: AgentCtx) -> Reader:
            if "reader" not in ctx.state:
                ctx.state["reader"] = Reader(doc, self.read_tokens)
            return ctx.state["reader"]

        sec_arg = {"type": ["integer", "string"], "description": f"Section number, 1 to {doc.n} (or Section<n>)."}
        return [
            ToolSpec("search_document", f"Keyword (BM25) search over the document's sections. Returns the {TOP_K} "
                     f"best-matching sections with a short snippet each.",
                     {"type": "object", "properties": {"query": {**STR, "description": "Search keywords."}},
                      "required": ["query"]}, lambda ctx, a: rd(ctx).search(str(a.get("query", "")))),
            ToolSpec("read_section", f"Returns the text of one section (up to about {self.read_tokens} tokens).",
                     {"type": "object", "properties": {"section": sec_arg}, "required": ["section"]},
                     lambda ctx, a: rd(ctx).read_section(a.get("section"))),
        ]

    def agent_state_summary(self, state: dict) -> dict | None:
        r = state.get("reader")
        return {"queries": r.queries, "read": r.read} if r else None

    def resource_exists(self, example: dict, run, ref: str) -> bool:
        m = SECTION_RE.match(ref)
        return bool(m) and 1 <= int(m.group(1)) <= self.doc(example).n

    def answer_properties(self):
        return {"answer": {"type": "string", "enum": list(LETTERS),
                           "description": "The letter of the correct option."}}, ["answer"]

    def score_agent(self, example: dict, final: dict | None) -> float:
        return float(norm_letter((final or {}).get("answer")) == example["answer"])

    def score_group(self, example: dict, finals: list[dict], run=None) -> dict:
        out = super().score_group(example, finals, run)
        out["preds"] = [None if f.get("failing") else norm_letter((f.get("final") or {}).get("answer")) for f in finals]
        return out
