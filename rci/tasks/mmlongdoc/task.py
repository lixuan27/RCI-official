"""MMLongBench-Doc: answer a question about one long PDF through page-level text and image tools."""
from __future__ import annotations

import base64
import json
import math
import random
import re
import threading
from collections import Counter
from pathlib import Path

from rci import paths
from rci.agents.tooling import STR, AgentCtx, ToolResult, ToolSpec
from rci.agents.tokens import count_tokens, truncate_to_tokens
from rci.tasks.base import Task

TOP_K, SNIPPET, READ_TOKENS, VIEW_BUDGET = 5, 200, 1500, 40
K1, B = 0.9, 0.4
TEXT_MIN_CHARS = 50
PAGE_RE = re.compile(r"^Page(\d+)$")
_WORD = re.compile(r"\w+")
_STOP = frozenset("a an and are as at be but by for if in into is it no not of on or such that the their then "
                  "there these they this to was will with what which who how many much does do did".split())

SYSTEM = """You answer a question about one long PDF document. You cannot see the document directly: you inspect it
page by page with tools. Pages are numbered Page1 to PageN in file order (Page1 is the first page of the file; the
page numbers printed on the pages may differ).

Tools:
- search_pages(query): keyword (BM25) search over the pages' text layers; returns the {top_k} best-matching pages with
  a short snippet each. Scanned or purely graphical pages have no text layer and never appear in results.
- read_page(page): the text layer of one page (up to about {read_tokens} tokens).
- view_page(page): the rendered image of one page. Use it for charts, tables, figures, layout, and pages without a
  text layer. You can view at most {budget} page images, so locate pages with search_pages / read_page first.
Images drop out of your context when your earlier work is summarized, so write down what you read off an image
before moving on.

Answers may need several pages, counting, or arithmetic over values in the document. Some questions cannot be
answered from the document; then the answer is "Not answerable". When you are confident, call `submit_answer` with
a short, exact answer: a number, a short string, or a list such as ["A", "B"]. Put no explanation in the answer."""
_VIEW = SYSTEM[SYSTEM.index("- view_page(page)"):SYSTEM.index("Answers may need")]
SYSTEM_TEXT = SYSTEM.replace(_VIEW, "\n")


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def clean_text(text: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def doc_stem(doc_id: str) -> str:
    return doc_id[:-4] if doc_id.lower().endswith(".pdf") else doc_id


class PageDoc:
    """One pre-rendered document: page texts, BM25 statistics, and image paths."""

    def __init__(self, root: Path, doc_id: str):
        d = root / "docs" / doc_stem(doc_id)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"{d}/meta.json missing: run scripts/data/prepare_mmlongdoc.py")
        self.doc_id, self.dir = doc_id, d
        self.meta = json.loads((d / "meta.json").read_text())
        self.texts = [clean_text(t) for t in json.loads((d / "text.json").read_text())]
        self.n = len(self.texts)
        self.has_text = [len(re.sub(r"\s", "", t)) >= TEXT_MIN_CHARS for t in self.texts]
        self.tf = [Counter(tokenize(t)) if ok else Counter() for t, ok in zip(self.texts, self.has_text)]
        self.dl = [sum(c.values()) for c in self.tf]
        n_text = sum(self.has_text)
        self.avgdl = (sum(self.dl) / n_text) if n_text else 1.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {w: math.log(1 + (n_text - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    @property
    def n_text(self) -> int:
        return sum(self.has_text)

    def no_text_pages(self) -> list[int]:
        return [i + 1 for i, ok in enumerate(self.has_text) if not ok]

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

    def snippet(self, page: int, query: str, width: int = SNIPPET) -> str:
        text = re.sub(r"\s+", " ", self.texts[page - 1])
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

    def image_data_uri(self, page: int) -> str:
        data = (self.dir / "img" / f"{page}.jpg").read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


_LOCK = threading.Lock()
_DOCS: dict[tuple[str, str], PageDoc] = {}


def shared_doc(root: str, doc_id: str) -> PageDoc:
    key = (root, doc_id)
    with _LOCK:
        if key not in _DOCS:
            if len(_DOCS) >= 64:
                _DOCS.clear()
            _DOCS[key] = PageDoc(Path(root), doc_id)
        return _DOCS[key]


def parse_page(v, n: int) -> tuple[int | None, str | None]:
    if isinstance(v, str):
        s = v.strip()
        m = PAGE_RE.match(s)
        s = m.group(1) if m else s
        if not s.isdigit():
            return None, f"Error: page must be a page number such as 12 or Page12, got {v!r}."
        v = int(s)
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, bool) or not isinstance(v, int):
        return None, f"Error: page must be a page number such as 12 or Page12, got {v!r}."
    if not 1 <= v <= n:
        return None, f"Error: Page{v} does not exist; this document has Page1 to Page{n}."
    return v, None


class Reader:
    """Per-agent tool state."""

    def __init__(self, doc: PageDoc, view_budget: int, read_tokens: int):
        self.d, self.view_budget, self.read_tokens = doc, view_budget, read_tokens
        self.queries: list[str] = []
        self.read: list[int] = []
        self.viewed: list[int] = []

    def search(self, query: str) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult("Error: empty query.", ok=False)
        self.queries.append(query)
        d = self.d
        note = (f"{len(d.no_text_pages())} of {d.n} pages have no text layer and are not searchable; use view_page "
                f"for them." if d.n_text < d.n else "")
        if d.n_text == 0:
            return ToolResult("This document has no text layer, so search_pages finds nothing; use view_page.")
        hits = d.search(query)
        if not hits:
            return ToolResult(f"No page's text layer matches \"{query}\". {note}".strip())
        lines = [f"Top {len(hits)} pages for \"{query}\":"]
        lines += [f"Page{p} [score {s:.2f}]: {d.snippet(p, query)}" for p, s in hits]
        if note:
            lines.append(f"({note})")
        return ToolResult("\n".join(lines))

    def read_page(self, page) -> ToolResult:
        p, err = parse_page(page, self.d.n)
        if err:
            return ToolResult(err, ok=False)
        if p not in self.read:
            self.read.append(p)
        if not self.d.has_text[p - 1]:
            return ToolResult(f"Page{p} has no text layer (scanned page or picture). Use view_page({p}) to see it.")
        text = self.d.texts[p - 1]
        n_tok = count_tokens(text)
        head = f"Page{p} of {self.d.n} (text layer):\n"
        if n_tok <= self.read_tokens:
            return ToolResult(head + text)
        return ToolResult(head + truncate_to_tokens(text, self.read_tokens)
                          + f"\n[truncated at {self.read_tokens} of {n_tok} tokens; view_page({p}) shows the whole page]")

    def view_page(self, page) -> ToolResult:
        p, err = parse_page(page, self.d.n)
        if err:
            return ToolResult(err, ok=False)
        if len(self.viewed) >= self.view_budget:
            return ToolResult(f"Error: you have used all {self.view_budget} page views; use read_page instead.",
                              ok=False)
        self.viewed.append(p)
        left = self.view_budget - len(self.viewed)
        return ToolResult(f"Page{p} of {self.d.n}: the page image follows. {left} page views left.",
                          content=[{"type": "text", "text": f"Image of Page{p}:"},
                                   {"type": "image_url", "image_url": {"url": self.d.image_data_uri(p)}}])


class MMLongBenchDoc(Task):
    name = "mmlongdoc"
    msg_token_limit = 256
    max_actions = 100
    max_steps = 120
    summarize = True

    def __init__(self, root: str = "", limit_default: int | None = None, subset_seed: int | None = None,
                 view_budget: int = VIEW_BUDGET, read_tokens: int = READ_TOKENS, judge_url: str | None = None,
                 judge_model: str | None = None, file: str = "samples.jsonl"):
        self.root = root or str(paths.data_dir("mmlongdoc"))
        self.limit_default, self.subset_seed = limit_default, subset_seed
        self.view_budget, self.read_tokens = view_budget, read_tokens
        self.judge_url, self.judge_model = judge_url, judge_model
        self.file = file
        self._judge = None
        self._rows: list[dict] | None = None

    def examples(self, limit: int | None = None) -> list[dict]:
        if self._rows is None:
            p = Path(self.root) / self.file
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: run scripts/data/prepare_mmlongdoc.py")
            with open(p) as f:
                self._rows = [json.loads(l) for l in f if l.strip()]
        rows = sorted(self._rows, key=lambda r: r["idx"])
        if self.subset_seed is not None:
            random.Random(self.subset_seed).shuffle(rows)
        n = limit or self.limit_default
        return rows[:n] if n else rows

    def doc(self, example: dict) -> PageDoc:
        return shared_doc(self.root, example["doc_id"])

    def system_prompt(self, example: dict) -> str:
        return (SYSTEM if self.view_budget else SYSTEM_TEXT).format(top_k=TOP_K, read_tokens=self.read_tokens,
                                                                      budget=self.view_budget)

    def task_prompt(self, example: dict, agent_id: int, n_agents: int, rng) -> str:
        d = self.doc(example)
        if d.n_text == d.n:
            cov = "every page has a text layer"
        elif d.n_text == 0:
            cov = "no page has a text layer, so view the pages" if self.view_budget else "no page has a text layer"
        else:
            cov = f"{d.n_text} pages have a text layer; the other {d.n - d.n_text} are images only"
        return (f"Document: {example['doc_id']} ({d.n} pages, Page1 to Page{d.n}; {cov}).\n\n"
                f"Question: {example['question']}")

    def tools(self, example: dict, run) -> list[ToolSpec]:
        doc = self.doc(example)

        def rd(ctx: AgentCtx) -> Reader:
            if "reader" not in ctx.state:
                ctx.state["reader"] = Reader(doc, self.view_budget, self.read_tokens)
            return ctx.state["reader"]

        page_arg = {"type": ["integer", "string"], "description": f"Page number, 1 to {doc.n} (or Page<n>)."}
        return [
            ToolSpec("search_pages", f"Keyword (BM25) search over the text layer of the document's pages. Returns the "
                     f"{TOP_K} best-matching pages with a short snippet each.",
                     {"type": "object", "properties": {"query": {**STR, "description": "Search keywords."}},
                      "required": ["query"]}, lambda ctx, a: rd(ctx).search(str(a.get("query", "")))),
            ToolSpec("read_page", f"Returns the text layer of one page (up to about {self.read_tokens} tokens).",
                     {"type": "object", "properties": {"page": page_arg}, "required": ["page"]},
                     lambda ctx, a: rd(ctx).read_page(a.get("page"))),
            ToolSpec("view_page", f"Shows the rendered image of one page (for charts, tables, figures, layout and "
                     f"pages without text). At most {self.view_budget} views per task.",
                     {"type": "object", "properties": {"page": page_arg}, "required": ["page"]},
                     lambda ctx, a: rd(ctx).view_page(a.get("page"))),
        ][: 3 if self.view_budget else 2]

    def agent_state_summary(self, state: dict) -> dict | None:
        r = state.get("reader")
        return {"queries": r.queries, "read": r.read, "viewed": r.viewed} if r else None

    def resource_exists(self, example: dict, run, ref: str) -> bool:
        m = PAGE_RE.match(ref)
        return bool(m) and 1 <= int(m.group(1)) <= self.doc(example).n

    def answer_properties(self):
        return {"answer": {**STR, "description": "Your short, exact final answer (or \"Not answerable\")."}}, ["answer"]

    def score_agent(self, example: dict, final: dict | None) -> float:
        raise RuntimeError("MMLongBench-Doc answers are graded post hoc by rci.tasks.mmlongdoc.grade")

    def score_group(self, example: dict, finals: list[dict], run=None):
        if self.judge_url:
            return self._judged(example, finals)
        return {"score": None, "pending_judge": True,
                "per_agent": [None if f.get("failing") else (0.0 if not f.get("final") else None) for f in finals]}

    async def _judged(self, example: dict, finals: list[dict]) -> dict:
        from rci.agents.llm import LLMClient
        from rci.tasks.mmlongdoc.eval_score import eval_score
        from rci.tasks.mmlongdoc.grade import FAILED, extraction_messages, parse_extraction, response_text

        if self._judge is None:
            self._judge = LLMClient(self.judge_url.split(","), self.judge_model)
        per, preds = [], []
        for f in finals:
            fin = f.get("final")
            if f.get("failing"):
                per.append(None); preds.append(None)
                continue
            resp = response_text(fin)
            if not resp:
                per.append(0.0); preds.append(None)
                continue
            call = await self._judge.chat(extraction_messages(example["question"], resp), max_tokens=256,
                                          temperature=0.0, enable_thinking=False, tag="judge")
            pred = parse_extraction(call.content or "")
            per.append(0.0 if pred == FAILED else float(eval_score(example["answer"], pred, example["answer_format"])))
            preds.append(None if pred == FAILED else pred)
        valid = [x for x in per if x is not None]
        return {"score": max(valid) if valid else 0.0, "per_agent": per, "preds": preds, "judge": self.judge_model}
