"""BrowseComp-Plus: deep-research questions over a fixed web corpus with BM25 search and page tools."""
from __future__ import annotations

import datetime as _dt
import json
import re
import textwrap
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from rci import paths
from rci.agents.tooling import INT, STR, AgentCtx, ToolResult, ToolSpec
from rci.agents.tokens import count_tokens
from rci.tasks.base import Task

DEVELOPER_CONTENT = """You are a helpful assistant and harmless assistant.

You will be able to use a set of browsering tools to answer user queries.

Tool for browsing.
The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
Cite information from the tool using the following format:
`\u3010{cursor}\u2020L{line_start}(-L{line_end})?\u3011`, for example: `\u30106\u2020L9-L11\u3011` or `\u30108\u2020L3\u3011`.
Do not quote more than 10 words directly from the tool output.
sources=web"""

SUBMIT_HINT = """When you have found the answer, call `submit_answer` with your exact final answer and a brief
explanation. Pages name each document as Doc<docid>; use these names to reference documents."""

_FM_PATTERN = re.compile(
    r"---\s*"
    r"(?:title:\s*(?P<title>.+?)\s*)"
    r"(?:author:\s*(?P<author>.+?)\s*)?"
    r"(?:date:\s*(?P<date>.+?)\s*)?"
    r"---\s*"
    r"(?P<content>.*?)(?=---|$)",
    re.DOTALL,
)
DOC_RE = re.compile(r"^Doc(\S+)$")
VIEW_TOKENS, WRAP, MAX_RESULTS, SNIPPET = 1024, 80, 20, 180

_LOCK = threading.Lock()
_SHARED: dict = {}


def parse_content(docid: str, raw: str) -> tuple[str, str]:
    m = next(iter(_FM_PATTERN.finditer(raw)), None)
    if m:
        return (m.group("title") or f"Doc {docid}").strip(), (m.group("content") or "").strip()
    return f"Doc {docid}", raw.strip()


class Corpus:
    """Docs read on demand via byte offsets; BM25 via the prebuilt bm25s index."""

    def __init__(self, root: Path):
        self.root = root
        off = root / "corpus_offsets.json"
        if not off.exists():
            self._build_offsets(off)
        meta = json.loads(off.read_text())
        self.offsets = meta["offsets"]
        self.url2id = meta["url2id"]
        self.titles = meta["titles"]
        self.f = open(root / "corpus.jsonl", "rb")
        self.flock = threading.Lock()
        self.cache: dict[str, dict] = {}
        self._searcher = None

    def _build_offsets(self, out: Path) -> None:
        offsets, url2id, titles = {}, {}, {}
        with open(self.root / "corpus.jsonl", "rb") as f:
            pos = 0
            for line in f:
                r = json.loads(line)
                offsets[r["docid"]] = [pos, len(line)]
                url2id[r["url"]] = r["docid"]
                titles[r["docid"]] = r.get("title") or ""
                pos += len(line)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({"offsets": offsets, "url2id": url2id, "titles": titles}))
        tmp.replace(out)

    def doc(self, docid: str) -> dict | None:
        if docid in self.cache:
            return self.cache[docid]
        o = self.offsets.get(docid)
        if o is None:
            return None
        with self.flock:
            self.f.seek(o[0])
            r = json.loads(self.f.read(o[1]))
        title, body = parse_content(docid, r["text"])
        d = {"docid": docid, "url": r["url"], "title": title, "body": body, "raw": r["text"]}
        if len(self.cache) > 4096:
            self.cache.clear()
        self.cache[docid] = d
        return d

    @property
    def searcher(self):
        if self._searcher is None:
            from rci.tasks.browsecomp.bm25 import Searcher
            self._searcher = Searcher(str(self.root / "index" / "bm25s"))
        return self._searcher


def shared_corpus(root: str) -> Corpus:
    with _LOCK:
        if root not in _SHARED:
            _SHARED[root] = Corpus(Path(root))
        return _SHARED[root]


def wrap_lines(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    return textwrap.wrap(flat, WRAP, replace_whitespace=False, drop_whitespace=True) or [""]


_WORD = re.compile(r"\w+")
_STOP = frozenset("a an and are as at be but by for if in into is it no not of on or such that the their then "
                  "there these they this to was will with".split())


def summary_for(body: str, query: str, analyzer=None) -> str:
    """Best-matching sentence up to 180 characters (cheap stand-in for the Lucene highlighter)."""
    q = {w for w in _WORD.findall(query.lower()) if w not in _STOP}
    best, best_s = "", -1
    for s in re.split(r"(?<=[.!?])\s+", body[:20000])[:200]:
        sc = len(q.intersection(_WORD.findall(s.lower())))
        if sc > best_s:
            best, best_s = s, sc
    return re.sub(r"\s+", " ", best or body)[:SNIPPET]


class Browser:
    """Per-agent page stack."""

    def __init__(self, corpus: Corpus):
        self.c = corpus
        self.pages: list[dict] = []
        self.opened: list[str] = []
        self.seen: set[str] = set()

    def _push(self, page: dict) -> int:
        self.pages.append(page)
        return len(self.pages) - 1

    def _view(self, cur: int, loc: int = 0, num_lines: int = -1) -> str:
        p = self.pages[cur]
        lines = p["lines"]
        if loc < 0:
            loc = 0
        if loc >= len(lines):
            self.pages.pop()
            return f"Invalid location parameter: `{loc}`. Cannot exceed page maximum of {len(lines) - 1}."
        if num_lines > 0:
            end = min(len(lines), loc + num_lines)
        else:
            end, used = loc, 0
            while end < len(lines):
                used += count_tokens(f"L{end}: {lines[end]}\n")
                end += 1
                if used >= VIEW_TOKENS:
                    break
        body = "\n".join(f"L{i}: {lines[i]}" for i in range(loc, end))
        doc = f" Doc{p['docid']}" if p.get("docid") else ""
        return (f"[{cur}] {p['title']} ({p['url']}){doc}\n**viewing lines [{loc} - {end - 1}] of "
                f"{len(lines) - 1}**\n\n{body}")

    def search(self, query: str) -> str:
        hits = self.c.searcher.search(query, k=MAX_RESULTS)
        if not hits:
            return "No results returned for any query"
        lines, links = ["", f"URL: web-search://ts={int(time.time())}", "# Search Results"], {}
        for i, (docid, _) in enumerate(hits):
            d = self.c.doc(docid)
            if d is None:
                continue
            summ = summary_for(d["body"], query)
            links[i] = {"docid": docid, "url": d["url"], "summary": summ}
            self.seen.add(docid)
            entry = f"  * \u3010{i}\u2020{d['title']}\u2020{urlparse(d['url']).netloc}\u3011 Doc{docid} {summ}"
            lines.extend(textwrap.wrap(entry, WRAP, subsequent_indent="    "))
        cur = self._push({"title": query, "url": lines[1][5:], "lines": lines, "links": links, "kind": "search"})
        return self._view(cur)

    def _open_doc(self, docid: str, loc: int, num_lines: int, near: str | None = None) -> str:
        d = self.c.doc(docid)
        if d is None:
            return f"Error: document Doc{docid} not found."
        if docid not in self.opened:
            self.opened.append(docid)
        lines = ["", f"URL: {d['url']}"] + wrap_lines(d["body"])
        if loc < 0 and near:
            key = re.sub(r"\s+", " ", near)[:40]
            joined = [i for i, ln in enumerate(lines) if key[:20] and key[:20] in ln]
            loc = max(0, joined[0] - 4) if joined and joined[0] > 4 else 0
        cur = self._push({"title": d["title"], "url": d["url"], "docid": docid, "lines": lines, "links": {},
                          "kind": "doc"})
        return self._view(cur, max(loc, 0), num_lines)

    def open(self, id=-1, cursor: int = -1, loc: int = -1, num_lines: int = -1) -> str:
        if not self.pages and (id == -1 or isinstance(id, int)):
            return "Error: no page is open; search first or pass a URL."
        if isinstance(id, str) and not id.lstrip("-").isdigit():
            docid = self.c.url2id.get(id) or (id[3:] if id.startswith("Doc") else None)
            if docid is None or docid not in self.c.offsets:
                return f"Error: unknown URL `{id}`."
            return self._open_doc(docid, loc, num_lines)
        id = int(id)
        cur = cursor if 0 <= cursor < len(self.pages) else len(self.pages) - 1
        page = self.pages[cur]
        if id == -1:
            new = self._push(dict(page))
            return self._view(new, max(loc, 0), num_lines)
        link = page.get("links", {}).get(id)
        if link is None:
            return f"Invalid link id `{id}`."
        return self._open_doc(link["docid"], loc, num_lines, near=link.get("summary"))

    def find(self, pattern: str, cursor: int = -1) -> str:
        if not self.pages:
            return "Error: no page is open."
        cur = cursor if 0 <= cursor < len(self.pages) else len(self.pages) - 1
        page = self.pages[cur]
        if page["kind"] != "doc":
            return "Error: find cannot be used on search results or find results."
        p = pattern.lower()
        out, k = [], 0
        for i, ln in enumerate(page["lines"]):
            if p in ln.lower():
                out.append(f"# \u3010{k}\u2020match at L{i}\u3011")
                out.extend(page["lines"][i:i + 4])
                k += 1
                if k >= 50:
                    break
        if not out:
            return f"No `find` results for pattern: `{pattern}`"
        new = self._push({"title": f"Find results for text: `{pattern}` in `{page['title']}`", "url": page["url"],
                          "docid": page.get("docid"), "lines": [""] + out, "links": {}, "kind": "find"})
        return self._view(new)


class BrowseCompPlus(Task):
    name = "browsecomp"
    msg_token_limit = 256
    max_actions = 300
    max_steps = 400
    summarize = True

    def __init__(self, root: str = "", limit_default: int = 100, split: str = "test", offset: int = 0,
                 judge_url: str | None = None, judge_model: str | None = None):
        self.root = root or str(paths.data_dir("browsecomp"))
        self.limit_default, self.offset = limit_default, offset
        self.judge_url, self.judge_model = judge_url, judge_model
        self._judge = None
        self._queries = None

    def examples(self, limit: int | None = None) -> list[dict]:
        if self._queries is None:
            with open(Path(self.root) / "queries.jsonl") as f:
                self._queries = [json.loads(l) for l in f if l.strip()]
        qs = sorted(self._queries, key=lambda q: q["idx"])[self.offset: self.offset + (limit or self.limit_default)]
        return [{"id": f"q{q['query_id']}", **q} for q in qs]

    def system_prompt(self, example: dict) -> str:
        return DEVELOPER_CONTENT + "\n\nToday's date: " + _dt.date.today().isoformat() + "\n\n" + SUBMIT_HINT

    def task_prompt(self, example: dict, agent_id: int, n_agents: int, rng) -> str:
        return example["question"]

    def tools(self, example: dict, run) -> list[ToolSpec]:
        corpus = shared_corpus(self.root)

        def br(ctx: AgentCtx) -> Browser:
            if "browser" not in ctx.state:
                ctx.state["browser"] = Browser(corpus)
            return ctx.state["browser"]

        def search(ctx, args):
            return ToolResult(br(ctx).search(str(args.get("query", ""))))

        def open_(ctx, args):
            idv = args.get("id", -1)
            if isinstance(idv, str) and idv.strip().lstrip("-").isdigit():
                idv = int(idv)
            return ToolResult(br(ctx).open(idv, int(args.get("cursor", -1)), int(args.get("loc", -1)),
                                           int(args.get("num_lines", -1))))

        def find(ctx, args):
            return ToolResult(br(ctx).find(str(args.get("pattern", "")), int(args.get("cursor", -1))))

        return [
            ToolSpec("browser_search", "Searches for information related to a query and displays top N results. "
                     "Returns a list of search results with titles, URLs, and summaries.",
                     {"type": "object", "properties": {"query": {**STR, "description": "The search query string"}},
                      "required": ["query"]}, search),
            ToolSpec("browser_open", "Opens a link from the current page or a fully qualified URL. Can scroll to a "
                     "specific location and display a specific number of lines. Valid link ids are displayed with "
                     "the formatting: \u3010{id}\u2020.*\u3011.",
                     {"type": "object", "properties": {
                         "id": {"type": ["integer", "string"], "description": "Link id from current page (integer) or "
                                "fully qualified URL (string). Default is -1 (most recent page)"},
                         "cursor": {**INT, "description": "Page cursor to operate on. If not provided, the most recent page is implied"},
                         "loc": {**INT, "description": "Starting line number. If not provided, viewport will be positioned at the beginning or centered on relevant passage"},
                         "num_lines": {**INT, "description": "Number of lines to display"}}, "required": []}, open_),
            ToolSpec("browser_find", "Finds exact matches of a pattern in the current page or a specified page by cursor.",
                     {"type": "object", "properties": {
                         "pattern": {**STR, "description": "The exact text pattern to search for"},
                         "cursor": {**INT, "description": "Page cursor to search in. If not provided, searches in the current page"}},
                      "required": ["pattern"]}, find),
        ]

    def agent_state_summary(self, state: dict) -> dict | None:
        b = state.get("browser")
        return {"opened": b.opened, "seen": sorted(b.seen)} if b else None

    def resource_exists(self, example: dict, run, ref: str) -> bool:
        c = shared_corpus(self.root)
        m = DOC_RE.match(ref)
        return (bool(m) and m.group(1) in c.offsets) or ref in c.url2id

    def answer_properties(self):
        return {"answer": {**STR, "description": "Your exact final answer."},
                "explanation": {**STR, "description": "Brief explanation of how the evidence supports it."}}, ["answer"]

    def score_agent(self, example: dict, final: dict | None) -> float:
        raise RuntimeError("BrowseComp-Plus answers are graded post hoc by rci.tasks.browsecomp.grade")

    def score_group(self, example: dict, finals: list[dict], run=None):
        if self.judge_url:
            return self._judged(example, finals)
        return {"score": None, "pending_judge": True,
                "per_agent": [None if f.get("failing") else (0.0 if not f.get("final") else None) for f in finals]}

    async def _judged(self, example: dict, finals: list[dict]) -> dict:
        from rci.agents.llm import LLMClient
        from rci.tasks.browsecomp.grade import GRADER, _CORRECT, response_text

        if self._judge is None:
            self._judge = LLMClient(self.judge_url.split(","), self.judge_model)
        q = example.get("question") or example.get("query")
        per = []
        for f in finals:
            fin = f.get("final")
            if f.get("failing"):
                per.append(None)
            elif not fin or not str(fin.get("answer") or "").strip():
                per.append(0.0)
            else:
                call = await self._judge.chat([{"role": "user", "content": GRADER.format(
                    question=q, response=response_text(fin), correct_answer=example["answer"])}],
                    max_tokens=2048, temperature=0.0, enable_thinking=False, tag="judge")
                m = _CORRECT.search(call.content or "")
                per.append(1.0 if m and m.group(1).lower() == "yes" else 0.0)
        valid = [x for x in per if x is not None]
        return {"score": max(valid) if valid else 0.0, "per_agent": per, "judge": self.judge_model}
