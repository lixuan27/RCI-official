"""DocSweep: questions whose answer is spread over a shared corpus no single member can read in full."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from rci import paths
from rci.agents.tooling import AgentCtx, ToolResult, ToolSpec
from rci.tasks.base import Task
from rci.tasks.docsweep import normalise as N
from rci.tasks.docsweep.generate import corpus_sha256, render_doc

try:
    from rci.agents import events as _events
except ImportError:
    _events = None

LONE_BLOCKS = ("notes", "opened")
GROUP_BLOCKS = ("notes", "opened", "board", "claims", "analysis")
MAP_INTEGRATOR_BLOCKS = ("notes", "opened", "board")
BLOCK_PHRASES = {"notes": "your notes", "opened": "the numbers of the documents you have opened",
                 "board": "the digest board", "claims": "the claim board", "analysis": "the first analysis"}
MEMBER_SENTENCE = "The answer depends on every document; you cannot read them all within your own action limit."
CAPS = ("You have {actions} actions and {steps} rounds; each tool call uses one action unless its description says it "
        "is free.")
COMPACT_AT = 28672
DEFAULT_ACTIONS, DEFAULT_STEPS = 12, 24
NOTE_CAP = 6000
NOTE_ADVICE = ("As you read each document, copy into your notes, word for word, every figure, date and unit the question "
               "may need from it: when your history is compacted, only your notes are kept.")
SEARCH_HITS, HIT_CHARS, HIT_TOKENS, HIT_LEAD = 8, 160, 48, 40
OPEN_PER_TURN, SEARCH_PER_TURN = 2, 4
QUOTE_CHARS = (20, 300)
QUOTES_MAX, DIGEST_CHARS, DIGEST_TOKENS, DIGEST_RENDERED = 8, 1800, 400, 450
ANALYSIS_TOKENS = 1500
ORACLE_PROMPT_CAP = 36864
ACCESS = ("all", "tree")
SPLITS = ("eval", "smoke", "train", "k8", "all")
DEFAULT_QUESTIONS = "configs/docsweep/questions.json"

ENV = ("You answer a question about a corpus of {k} documents, numbered 1 to {k}. Each document is about one entity. "
       "The answer depends on figures written inside the documents: you have to compare dated readings, add up parts "
       "and convert units yourself.\n\n"
       "Tools:\n"
       "- `open_doc(doc)`: the full text of one document. Uses one action; at most 2 calls per round.\n"
       "- `search(pattern)`: the lines that contain a piece of text (a literal match, ignoring case); at most 8 hits, "
       "each a document number and a cut of the line. Uses one action; at most 4 calls per round.\n"
       "- `note(text)`: adds a note to your private notes. Free: it uses no action.\n"
       "- `submit_answer(answer)`: submits your final answer and ends your work.\n"
       "Each of your turns is one round (a step). Every tool result ends with a line giving the actions and steps you "
       "have left.")
ORACLE_ENV = ("You answer a question about a corpus of {k} documents, all given in full after the question. Each "
              "document is about one entity. The answer depends on figures written inside the documents: you have to "
              "compare dated readings, add up parts and convert units yourself. When you have worked it out, call "
              "`submit_answer` with your answer.")
ANSWER_FORMAT = {"agg": "every entity that satisfies the conditions, as a list of names (order does not matter)",
                 "join": "two names [A, B], in that order"}
ANSWER_DESCRIPTION = {"agg": "Every entity that satisfies the conditions, as a list of names (order does not matter).",
                      "join": "Two names [A, B], in that order."}


class CorpusOverflow(Exception):
    pass


def compaction_rule(blocks=LONE_BLOCKS, limit: int = COMPACT_AT) -> str:
    kept = ", ".join(BLOCK_PHRASES[b] for b in GROUP_BLOCKS if b in blocks)
    return (f"When your context reaches {limit:,} tokens, your history keeps only {kept} and the results of your last "
            f"turn's reads; your thinking from the last turn is not kept. Record with `note` any value you will need.")


def render_digest(doc: int, author: int, quotes: list[str]) -> str:
    """A verified digest as every reader sees it (the 450-token check, injections and the board block)."""
    return (f"Digest of document {doc} (verified quotes, posted by Participant {author}):\n"
            + "\n".join(f'- "{q}"' for q in quotes))


def as_doc(x) -> int | None:
    """A document number from a tool argument: an int, an integral float, or a string such as "7" or "doc 7"."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        m = re.fullmatch(r"\s*(?:doc(?:ument)?\s*#?\s*)?(\d+)\s*", x, re.I)
        return int(m.group(1)) if m else None
    return None


def seat_docs(k: int, seat: int) -> set[int]:
    seat %= 4
    return set() if seat == 0 else {d for d in range(1, k + 1) if (d - 1) % 3 == seat - 1}


def _ranges(docs) -> str:
    return ", ".join(str(d) for d in sorted(docs)) or "none"


def hit_window(line: str, start: int, end: int, tok) -> str:
    def cut(text: str) -> str:
        return tok.truncate(text[:HIT_CHARS], HIT_TOKENS)
    w = cut(line)
    if end <= len(w):
        return w
    ws = max(0, start - HIT_LEAD)
    while 0 < ws < start and line[ws - 1].isalnum():
        ws += 1
    while ws < start and line[ws].isspace():
        ws += 1
    return cut(line[ws:])


def search_hits(example: dict, pattern: str, tok, allowed=None, limit: int = SEARCH_HITS) -> list[tuple]:
    rx = re.compile(re.escape(pattern), re.I)
    out = []
    for doc in example["docs"]:
        if allowed is not None and doc["doc"] not in allowed:
            continue
        for li, line in enumerate(doc["text"].split("\n")):
            for m in rx.finditer(line):
                out.append((doc["doc"], li, m.start(), hit_window(line, m.start(), m.end(), tok)))
                if len(out) >= limit:
                    return out
    return out


def _spec(name: str, description: str, parameters: dict, fn, terminal: bool = False, **settings) -> ToolSpec:
    spec = ToolSpec(name, description, parameters, fn, terminal=terminal)
    for k, v in settings.items():
        setattr(spec, k, v)
    return spec


def _result(text: str, ok: bool = True, free: bool = False, meta: dict | None = None) -> ToolResult:
    res = ToolResult(text, ok=ok)
    res.free = free
    res.meta = meta
    return res


def _ds(state: dict) -> dict:
    st = state.get("ds")
    if st is None:
        st = state["ds"] = {"open_log": [], "opened": [], "notes": [], "notes_log": [], "note_drops": 0,
                            "submitted_step": None}
    return st


def _step(run, ctx: AgentCtx | None = None) -> int:
    s = getattr(run, "step", None)
    return int(s) if s is not None else int(getattr(ctx, "step", 0) or 0)


def next_seq(run) -> int:
    """The run-wide event sequence number, shared with the protocol's items."""
    run.state.setdefault("event_seq", 0)
    if _events is not None:
        return _events.next_seq(run)
    run.state["event_seq"] += 1
    return run.state["event_seq"]


def log_event(run, ctx: AgentCtx, op: str, **fields) -> dict | None:
    """Appends one event to the agent's log when the run keeps one."""
    if run.state.get("event_log") is not True:
        return None
    if _events is not None and getattr(run, "agents", None) and ctx.agent_id < len(run.agents):
        return _events.log(run, ctx.agent_id, op, **fields)
    ev = {"seq": next_seq(run), "op": op, "step": _step(run, ctx), "agent": ctx.agent_id, **fields}
    ctx.state.setdefault("events", []).append(ev)
    return ev


def _note_line(entry: dict) -> str:
    return f"- [round {entry['step'] + 1}] {entry['text']}"


class DocSweep(Task):
    name = "docsweep"
    max_actions = DEFAULT_ACTIONS
    max_steps = DEFAULT_STEPS
    summarize = False
    deliberate_first = False
    status_line = True
    meter_prompts = True
    deliver_on_final = True
    event_log = True

    def __init__(self, root: str = "", questions: str | None = None, split: str = "eval",
                 tokenizer=None, access: str = "all", compact_at: int = COMPACT_AT, note_advice: bool = False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if access not in ACCESS:
            raise ValueError(f"access must be one of {ACCESS}")
        self.root, self.questions, self.split = root or str(paths.data_dir("docsweep")), questions, split
        self.tokenizer, self.access, self.compact_at = tokenizer, access, int(compact_at)
        self.note_advice = bool(note_advice)
        self._tok = None
        self._q: dict | None = None
        self._rows: dict = {}
        self._truth: dict | None = None

    def chat_tokenizer(self):
        """The chat tokenizer every DocSweep token count uses: load_tokenizer(self.tokenizer)."""
        if self._tok is None:
            if self.tokenizer is None:
                raise ValueError("DocSweep needs tokenizer='stub' or a model directory")
            if isinstance(self.tokenizer, str):
                from rci.agents.tokenizer import load_tokenizer
                self._tok = load_tokenizer(self.tokenizer)
            else:
                self._tok = self.tokenizer
        return self._tok

    def _questions(self) -> dict:
        if self._q is None:
            if not self.questions and Path(DEFAULT_QUESTIONS).exists():
                self.questions = DEFAULT_QUESTIONS
            if not self.questions:
                raise ValueError(f"split {self.split!r} needs `questions` ({DEFAULT_QUESTIONS})")
            self._q = json.loads(Path(self.questions).read_text(encoding="utf-8"))
        return self._q

    def ids(self) -> list[str]:
        """The corpus ids of this split: every corpus in root for "all" (admission), else the questions file's list."""
        if self.split == "all":
            return sorted(p.stem for p in (Path(self.root) / "corpora").glob("*.json"))
        return list(self._questions().get(self.split) or [])

    def load(self, cid: str) -> dict:
        """One corpus; against the questions file's frozen sha256 when it records one (a changed corpus stops the run)."""
        if cid not in self._rows:
            ex = json.loads((Path(self.root) / "corpora" / f"{cid}.json").read_text(encoding="utf-8"))
            want = ((self._questions() if self.questions else {}).get("corpus_sha256") or {}).get(cid)
            if want is not None and corpus_sha256(ex) != want:
                raise ValueError(f"corpus {cid} does not match its frozen sha256")
            self._rows[cid] = ex
        return self._rows[cid]

    def examples(self, limit: int | None = None) -> list[dict]:
        ids = self.ids()
        return [self.load(i) for i in (ids[:limit] if limit else ids)]

    def truth(self, example_id: str) -> dict:
        """The sealed truth entry of one corpus (answer, latent table, needed values, decoy); never put in a prompt."""
        if self._truth is None:
            path = Path(self.root) / "truth.json"
            raw = path.read_bytes()
            want = (self._questions() if self.questions else {}).get("truth_sha256")
            if want is not None and hashlib.sha256(raw).hexdigest() != want:
                raise ValueError("truth.json does not match its frozen sha256")
            self._truth = json.loads(raw.decode("utf-8"))
        return self._truth[example_id]

    def system_prompt(self, example: dict, *, kind: str = "member", blocks=LONE_BLOCKS, actions: int | None = None,
                      steps: int | None = None) -> str:
        k = example["k"]
        if kind == "oracle":
            return ORACLE_ENV.format(k=k)
        if kind not in ("member", "solo"):
            raise ValueError(f"unknown kind {kind!r}")
        parts = [ENV.format(k=k), CAPS.format(actions=actions or DEFAULT_ACTIONS, steps=steps or DEFAULT_STEPS)]
        if self.note_advice:
            parts.append(NOTE_ADVICE)
        if kind == "member":
            parts.append(MEMBER_SENTENCE)
        parts.append(compaction_rule(blocks, self.compact_at))
        return "\n\n".join(parts)

    def answer_format(self, example: dict) -> str:
        return f"Answer format: {ANSWER_FORMAT[example['family']]}."

    def task_prompt(self, example: dict, agent_id: int, n_agents: int, rng) -> str:
        k = example["k"]
        return (f"{example['question']}\n\nThe corpus has {k} documents, numbered 1 to {k}.\n\n"
                f"{self.answer_format(example)}")

    def oracle_prompt(self, example: dict) -> str:
        """The full-information oracle's prompt: the question, then every document as `open_doc` returns it, in order."""
        k = example["k"]
        docs = "\n\n".join(render_doc(d["doc"], k, d["text"]) for d in example["docs"])
        return (f"{example['question']}\n\nThe corpus has {k} documents, numbered 1 to {k}; all of them follow.\n\n"
                f"{self.answer_format(example)}\n\n{docs}")

    def render_doc(self, example: dict, d: int) -> str:
        return render_doc(d, example["k"], example["docs"][d - 1]["text"])

    def allowed_docs(self, example: dict, agent_id: int) -> set[int] | None:
        return None if self.access == "all" else seat_docs(example["k"], agent_id)

    def tools(self, example: dict, run) -> list[ToolSpec]:
        k = example["k"]
        tok = self.chat_tokenizer()
        cap = getattr(getattr(run, "method", None), "note_cap", None) or NOTE_CAP

        def open_doc(ctx: AgentCtx, args: dict) -> ToolResult:
            raw = args.get("doc")
            d = as_doc(raw)
            if d is None or not 1 <= d <= k:
                return _result(f"Error: there is no document {raw!r}; the documents are numbered 1 to {k}.", ok=False,
                               meta={"docs": [], "verdict": "invalid_doc"})
            allowed = self.allowed_docs(example, ctx.agent_id)
            if allowed is not None and d not in allowed:
                return _result(f"Refused: from your seat you can open only documents {_ranges(allowed)}.", ok=False,
                               meta={"docs": [], "verdict": "not_allowed"})
            run_ = ctx.run
            st = _ds(ctx.state)
            st["open_log"].append({"doc": d, "step": _step(run_, ctx), "seq": next_seq(run_)})
            if d not in st["opened"]:
                st["opened"].append(d)
            return _result(self.render_doc(example, d), meta={"docs": [d], "verdict": "ok"})

        def search(ctx: AgentCtx, args: dict) -> ToolResult:
            p = args.get("pattern")
            p = "" if p is None else p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)
            if not p.strip():
                return _result("Error: the search pattern is empty.", ok=False,
                               meta={"docs": [], "verdict": "empty_pattern"})
            hits = search_hits(example, p, tok, self.allowed_docs(example, ctx.agent_id))
            if not hits:
                return _result(f'No hits for "{p}".', meta={"docs": [], "verdict": "no_hits"})
            docs = list(dict.fromkeys(h[0] for h in hits))
            return _result(f'{len(hits)} hit(s) for "{p}":\n' + "\n".join(f"Doc {d}: {w}" for d, _, _, w in hits),
                           meta={"docs": docs, "verdict": "ok"})

        def note(ctx: AgentCtx, args: dict) -> ToolResult:
            run_ = ctx.run
            text = args.get("text")
            text = "" if text is None else text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
            if not text.strip():
                return _result("Refused: the note is empty. Nothing was noted and no action was used.", ok=False,
                               free=True, meta={"verdict": "empty"})
            st = _ds(ctx.state)
            step = _step(run_, ctx)
            entry = {"id": len(st["notes_log"]), "step": step, "seq": None, "text": text, "tokens": 0}
            entry["tokens"] = tok.count(_note_line(entry) + "\n")
            if entry["tokens"] > cap:
                log_event(run_, ctx, "note_refused", tokens=entry["tokens"], cap=cap)
                return _result(f"Refused: this note is {entry['tokens']:,} tokens, more than your notes can hold "
                               f"({cap:,}). Nothing was noted and no action was used.", ok=False, free=True,
                               meta={"verdict": "too_long"})
            entry["seq"] = next_seq(run_)
            st["notes"].append(entry)
            st["notes_log"].append(dict(entry))
            total = sum(e["tokens"] for e in st["notes"])
            dropped = []
            while total > cap:
                old = st["notes"].pop(0)
                total -= old["tokens"]
                dropped.append(old)
            log_event(run_, ctx, "note", tokens=entry["tokens"], total_tokens=total, entries=len(st["notes"]))
            msg = f"Noted. Your notes hold {total:,} of {cap:,} tokens."
            if dropped:
                ev = log_event(run_, ctx, "note_drop", dropped_entries=len(dropped),
                               dropped_tokens=sum(e["tokens"] for e in dropped), cap=cap, total_tokens=total)
                drop_seq = ev["seq"] if ev else next_seq(run_)
                ids = {e["id"] for e in dropped}
                for e in st["notes_log"]:
                    if e["id"] in ids:
                        e["dropped_seq"] = drop_seq
                st["note_drops"] += len(dropped)
                msg += f" The oldest {len(dropped)} note(s) were dropped to make room."
            return _result(msg, meta={"verdict": "ok"})

        doc_arg = {"type": "integer", "description": f"The document number, 1 to {k}."}
        return [
            _spec("open_doc", f"Returns the full text of one document. Uses one action; at most {OPEN_PER_TURN} calls "
                  f"per round.", {"type": "object", "properties": {"doc": doc_arg}, "required": ["doc"]}, open_doc,
                  per_turn=OPEN_PER_TURN),
            _spec("search", f"Finds the lines of the documents that contain `pattern` (a literal piece of text; case is "
                  f"ignored) and returns at most {SEARCH_HITS} hits, each the document number and a cut of the line of "
                  f"at most {HIT_CHARS} characters. Uses one action; at most {SEARCH_PER_TURN} calls per round.",
                  {"type": "object", "properties": {"pattern": {"type": "string", "description": "The text to find."}},
                   "required": ["pattern"]}, search, per_turn=SEARCH_PER_TURN),
            _spec("note", f"Adds `text` to your private notes, which you keep when your history is compacted. Free: it "
                  f"uses no action. Your notes hold at most {cap:,} tokens; when they are full, the oldest notes are "
                  f"dropped." + (" " + NOTE_ADVICE if self.note_advice else ""), {"type": "object", "properties": {"text": {"type": "string",
                                                                          "description": "What to note."}},
                                "required": ["text"]}, note, free=True, on_compact="effect"),
        ]

    def opened_seen(self, run, agent_id: int) -> set[int]:
        """Documents the agent has opened itself at an earlier step (its result has been in a prompt it sent)."""
        agent = run.agents[agent_id]
        st = agent.ctx.state.get("ds") or {}
        now = _step(run)
        return {e["doc"] for e in st.get("open_log") or [] if e["step"] < now}

    def covered(self, run, agent_id: int) -> set[int]:
        seen = (run.state.get("ds_seen_digests") or {}).get(agent_id) or ()
        return self.opened_seen(run, agent_id) | {int(d) for d in seen}

    def holds_every_fact(self, example: dict, run, author: int) -> bool:
        return set(range(1, example["k"] + 1)) <= self.covered(run, author)

    def certified(self, sub_task: str) -> bool:
        """Only the first analysis is vouched for by the coverage rule."""
        return sub_task == "analysis"

    def publish_validator(self, example: dict, run):
        k = example["k"]
        tok = self.chat_tokenizer()

        def check(author: int, doc, quotes) -> tuple[str, str]:
            if not isinstance(quotes, list) or not all(isinstance(q, str) for q in quotes) or \
                    not isinstance(doc, (int, float, str)) or isinstance(doc, bool):
                return "bad_format", ("a digest is a document number and a list of quotes, each a string copied word "
                                      "for word from that document")
            d = as_doc(doc)
            if d is None or not 1 <= d <= k:
                return "invalid_doc", f"there is no document {doc!r}; the documents are numbered 1 to {k}"
            if d not in self.opened_seen(run, author):
                return "not_opened", (f"you can post a digest only of a document you opened yourself in an earlier "
                                      f"round, and you had not opened document {d} before this round")
            if not quotes:
                return "no_quotes", "the digest has no quotes"
            if len(quotes) > QUOTES_MAX:
                return "too_many_quotes", f"at most {QUOTES_MAX} quotes per document"
            for i, q in enumerate(quotes, 1):
                if not QUOTE_CHARS[0] <= len(q) <= QUOTE_CHARS[1]:
                    return "quote_length", (f"quote {i} is {len(q)} characters; each quote must be "
                                            f"{QUOTE_CHARS[0]}-{QUOTE_CHARS[1]} characters")
            text = example["docs"][d - 1]["text"]
            for i, q in enumerate(quotes, 1):
                if q not in text:
                    return "not_verbatim", f"quote {i} is not copied word for word from document {d}: {q[:80]!r}"
            if sum(len(q) for q in quotes) > DIGEST_CHARS:
                return "too_long_chars", f"the quotes exceed {DIGEST_CHARS:,} characters in all"
            if sum(tok.count(q) for q in quotes) > DIGEST_TOKENS:
                return "too_long_tokens", f"the quotes exceed {DIGEST_TOKENS} tokens in all"
            if tok.count(render_digest(d, author, quotes)) > DIGEST_RENDERED:
                return "too_long_rendered", f"the digest with its header exceeds {DIGEST_RENDERED} tokens"
            return "ok", ""
        return check

    def analysis_error(self, example: dict, run, text) -> tuple[str, str] | None:
        if not isinstance(text, str) or not text.strip():
            return "empty", "The analysis is empty."
        n = self.chat_tokenizer().count(text)
        if n > ANALYSIS_TOKENS:
            return "too_long", f"The analysis is {n:,} tokens; at most {ANALYSIS_TOKENS:,} are allowed."
        return None

    @staticmethod
    def render_digest(doc: int, author: int, quotes: list[str]) -> str:
        return render_digest(doc, author, quotes)

    def memory_blocks(self, example: dict, run, agent) -> list[tuple[str, str]]:
        st = agent.ctx.state.get("ds") or {}
        lines = "\n".join(_note_line(e) for e in st.get("notes") or [])
        opened = ", ".join(str(d) for d in sorted(st.get("opened") or []))
        return [("notes", "## Your notes\n" + (lines or "(none yet)")),
                ("opened", "## Documents you have opened\n" + (opened or "(none yet)"))]

    def answer_properties(self):
        return {"answer": {"type": "array", "items": {"type": "string"},
                           "description": "The names of the entities in your answer."}}, ["answer"]

    def submit_tool(self, example: dict, with_refs: bool, validator=None, repair=None) -> ToolSpec:
        spec = super().submit_tool(example, with_refs, validator, repair)
        fam = example["family"]
        spec.parameters["properties"]["answer"] = {"type": "array", "items": {"type": "string"},
                                                   "description": ANSWER_DESCRIPTION[fam]}
        inner = spec.fn

        def fn(ctx: AgentCtx, args: dict) -> ToolResult:
            if "answer" in args:
                names = N.names_of(args.get("answer"))
                if fam == "join" and len(names) != 2 and not ctx.state.get("final_last_attempt"):
                    return ToolResult(f"Error: the answer must be exactly two names [A, B], in that order; got "
                                      f"{len(names)}.", ok=False)
                args = {**args, "answer": names}
            res = inner(ctx, args)
            if res.final is not None:
                _ds(ctx.state)["submitted_step"] = _step(ctx.run, ctx)
            return res

        spec.fn = fn
        return spec

    def score_agent(self, example: dict, final: dict | None) -> float:
        if not final:
            return 0.0
        truth = self.truth(example["id"])["answer"]
        got = [N.canon(example, n) for n in N.names_of(final.get("answer"))]
        if example["family"] == "join":
            return float(got == list(truth))
        return float(set(got) == set(truth))

    def score_group(self, example: dict, finals: list[dict], run=None) -> dict:
        out = super().score_group(example, finals, run)
        out["preds"] = [None if f.get("failing") or not f.get("final") else N.pred_of(example, f["final"].get("answer"))
                        for f in finals]
        out["family"] = example["family"]
        out["k"] = int(example["k"])
        return out

    def agent_state_summary(self, state: dict) -> dict:
        """The event log and the per-agent DocSweep state, kept in each agent's record (`task_state`)."""
        return {"events": state.get("events", []), "ds": state.get("ds", {})}
