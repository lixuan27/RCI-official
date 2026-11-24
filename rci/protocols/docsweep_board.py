"""DocSweep group board: document claims, verified digests, the first analysis and incremental delivery."""
from __future__ import annotations

import hashlib
import re

from rci.agents.tooling import INT, STR, STR_LIST, AgentCtx, ToolResult, ToolSpec
from rci.protocols.base import Protocol
from rci.protocols.blackboard import BlackboardProtocol

LONE_BLOCKS = ("notes", "opened")
GROUP_BLOCKS = ("notes", "opened", "board", "claims", "analysis")
MAP_INTEGRATOR_BLOCKS = ("notes", "opened", "board")

MAX_CLAIMS = 3
MAX_CLAIM_IDS = 24
ANALYSIS_HEAD = "posted by a participant who has covered every document"
DIGEST_HEAD = "Digest of document {doc} (verified quotes, posted by Participant {author}):"

DIGEST_RULES = ("Each digest names one document you have opened with `open_doc` and holds only quotes copied word for word "
                "from it: 1 to 8 quotes, each 20 to 300 characters, at most 1,800 characters and 400 tokens in all, and at "
                "most 450 tokens with the digest's header; no other text. The host checks every quote and refuses a digest "
                "that fails. The board keeps only the first verified digest of each document; a later digest of the same "
                "document is refused as a duplicate, at no cost.")
DS_PROTOCOL = ("You are Participant {me} of a group of {n} participants (numbered 0 to {last}) working on the same question "
               "with the same corpus; every participant can open every document. Each participant submits its own answer; "
               "the group commits to the answer the most members give, ties broken by their stated confidence.\n\n"
               "Group tools:\n"
               "{claim}"
               "- `post_digest`: post digests of documents you have opened, for the others. One action per call. "
               + DIGEST_RULES + "\n"
               "- `post_analysis`: post an analysis or recommendation in free text, at most 1,500 tokens (a longer one is "
               "refused, at no cost). One action. You can post one only after you have covered every document: opened it "
               "yourself, or received its verified digest. Only the first analysis in the group is accepted; it is delivered "
               "to everyone under the heading \"" + ANALYSIS_HEAD + "\". A later one is refused, at no cost.\n\n"
               "Delivery: {what} delivered once to every other participant, at its next call, as a message kept in its "
               "history.")
DS_ALL_ACCESS = "every participant can open every document"
DS_SEAT_ACCESS = ("each participant can open and search only the documents its seat's instructions name (Participant 0 "
                  "none)")
DS_CLAIM_DOC = ("- `claim`: tell the others which documents you will read. Free (it uses no action); at most {c} claims per "
                "participant, each of 1 to {m} distinct document numbers.\n")
DS_WAIT_NOTE = (" Timing rule: a participant that has not covered every document cannot submit until the first analysis has "
                "reached it, or until round {k}; a held submission costs no action.")
DS_PLAN_NOTE = " Reading opens at round {k}: before that round, `open_doc` and `search` are refused, at no cost.{claim}"
DS_PLAN_CLAIM = " Use the first rounds to plan and to claim documents."
DS_PLAN_HOLD = "Not yet: reading opens at round {k}. This refusal costs no action.{claim}"
DS_PLAN_HOLD_CLAIM = " Until then you can claim the documents you will read with `claim`."
DS_HOLD = ("Not yet: you have not covered every document (opened it yourself, or received its verified digest), and no "
           "analysis by a participant who has covered every document has reached you. Your answer is held until one of "
           "these happens, or until round {k}. This refusal costs no action: keep reading, or end your turn without calling "
           "a tool to wait.")
DS_COVER_REFUSED = ("Refused: only a participant that has covered every document (opened it yourself, or received its "
                    "verified digest) can post an analysis; you have covered {c} of {k}. This call used an action.")
DS_INJECT_HEAD = "New from the other participants since your last call:"


def _events():
    try:
        from rci.agents import events
    except ImportError:
        return None
    return events


def ds_seq(run) -> int:
    """The next run-wide sequence number (an item's `seq` comes from events.next_seq)."""
    run.state.setdefault("event_seq", 0)
    ev = _events()
    if ev is not None:
        return ev.next_seq(run)
    run.state["event_seq"] += 1
    return run.state["event_seq"]


def ds_event(run, agent_id: int, op: str, **fields) -> dict | None:
    ev = _events()
    if ev is not None:
        return ev.log(run, agent_id, op, **fields)
    if run.state.get("event_log") is not True:
        return None
    e = {"seq": ds_seq(run), "op": op, "step": getattr(run, "step", None), "agent": agent_id, **fields}
    run.agents[agent_id].ctx.state.setdefault("events", []).append(e)
    return e


def ds_tool(name: str, description: str, parameters: dict, fn, *, terminal: bool = False, free: bool = False,
            per_turn: int = 0, on_compact: str = "keep") -> ToolSpec:
    spec = ToolSpec(name, description, parameters, fn, terminal=terminal)
    spec.free, spec.per_turn, spec.on_compact = free, per_turn, on_compact
    return spec


def ds_result(text: str, ok: bool = True, *, free: bool = False, meta: dict | None = None,
              final: dict | None = None) -> ToolResult:
    res = ToolResult(text, ok=ok, final=final)
    res.free, res.meta = free, meta
    return res


def n_docs(run) -> int:
    """K, the corpus size of the run's example."""
    ex = run.example
    return int(ex.get("k") or len(ex.get("docs") or []))


def as_doc(x) -> int | None:
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


def doc_ranges(docs) -> str:
    """Sorted document numbers as compact ranges (consecutive numbers joined by a dash)."""
    xs = sorted(set(int(d) for d in docs))
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ", ".join(out)


def render_digest(run, doc: int, author: int, quotes: list[str]) -> str:
    """The one digest renderer: the task's own `render_digest`, else the default format."""
    fn = getattr(run.task, "render_digest", None)
    if fn is None:
        import importlib
        fn = getattr(importlib.import_module(type(run.task).__module__), "render_digest", None)
    if fn is not None:
        return fn(doc, author, quotes)
    return DIGEST_HEAD.format(doc=doc, author=author) + "\n" + "\n".join(f'- "{q}"' for q in quotes)


def covered_count(run, agent_id: int) -> int | None:
    fn = getattr(run.task, "covered", None)
    return len(fn(run, agent_id)) if fn is not None else None


def ds_init(run, n: int) -> None:
    st = run.state
    st["ds_items"], st["ds_index"] = [], {}
    st["ds_pending"] = {i: [] for i in range(n)}
    st["ds_offer"], st["ds_delivered"] = {}, {i: set() for i in range(n)}
    st["ds_board"], st["ds_claims"], st["ds_analysis"] = {}, [], None
    st["ds_seen_digests"] = {i: set() for i in range(n)}
    st["ds_analysis_seen"] = set()
    st["ds_refusals"] = []
    st.setdefault("holds", [])


def ds_new_item(run, kind: str, item_id: str, author: int, recipients, **extra) -> dict:
    item = {"id": item_id, "kind": kind, "author": author, "step": run.step, "seq": ds_seq(run), **extra}
    run.state["ds_items"].append(item)
    run.state["ds_index"][item_id] = item
    for j in recipients:
        if j != author:
            run.state["ds_pending"].setdefault(j, []).append(item_id)
    return item


def ds_refuse(run, agent_id: int, tool: str, verdict: str, doc=None) -> None:
    r = {"step": run.step, "agent": agent_id, "tool": tool, "verdict": verdict}
    if doc is not None:
        r["doc"] = doc
    r["seq"] = ds_seq(run)
    run.state["ds_refusals"].append(r)


def ds_hold(run, agent, kind: str, tool: str | None, **extra) -> None:
    """Log a hold (op `hold`) and keep it in run.state["holds"]."""
    run.state.setdefault("holds", []).append({"step": run.step, "agent": agent.id, "kind": kind, "tool": tool, **extra})
    ds_event(run, agent.id, "hold", kind=kind, tool=tool, **extra)


def render_items(run, items: list[dict], head: str) -> str:
    parts = [head]
    claims = [it for it in items if it["kind"] == "claim"]
    digests = [it for it in items if it["kind"] == "digest"]
    analysis = [it for it in items if it["kind"] == "analysis"]
    if claims:
        parts.append("## New claims\n" + "\n".join(
            f"Participant {it['author']} claimed documents {', '.join(str(d) for d in it['docs'])}." for it in claims))
    if digests:
        parts.append("## New verified digests\n" + "\n\n".join(
            render_digest(run, it["doc"], it["author"], it["quotes"]) for it in digests))
    if analysis:
        parts.append(f"## Analysis {ANALYSIS_HEAD}\n" + analysis[0]["text"])
    return "\n\n".join(parts)


def ds_offer(run, agent, head: str = DS_INJECT_HEAD) -> str | None:
    pend = run.state["ds_pending"].get(agent.id) or []
    if not pend:
        run.state["ds_offer"].pop(agent.id, None)
        return None
    idx = run.state["ds_index"]
    items = sorted((idx[i] for i in pend), key=lambda it: it["seq"])
    run.state["ds_offer"][agent.id] = [it["id"] for it in items]
    ds_event(run, agent.id, "offer", items=[it["id"] for it in items], call=len(agent.calls))
    return render_items(run, items, head)


def ds_confirm(run, agent, call: int, via: str, rebuilt: list[str]) -> None:
    offer = run.state["ds_offer"].pop(agent.id, None) or []
    ids = list(offer) if via in ("injection", "frozen_history") else list(rebuilt) if via == "rebuilt" else []
    idx = run.state["ds_index"]
    delivered = run.state["ds_delivered"].setdefault(agent.id, set())
    pend = run.state["ds_pending"].setdefault(agent.id, [])
    for iid in sorted(ids, key=lambda i: idx[i]["seq"]):
        it = idx[iid]
        if it["author"] == agent.id:
            continue
        ds_event(run, agent.id, "deliver", item=iid, kind=it["kind"], author=it["author"], via=via, call=call,
                 first=iid not in delivered)
        delivered.add(iid)
        if iid in pend:
            pend.remove(iid)
        if it["kind"] == "digest":
            run.state["ds_seen_digests"].setdefault(agent.id, set()).add(it["doc"])
        elif it["kind"] == "analysis":
            run.state["ds_analysis_seen"].add(agent.id)


def board_block(run) -> tuple[str, str]:
    """The rebuilt digest board: every verified digest in document order."""
    board = run.state["ds_board"]
    body = "\n\n".join(render_digest(run, d, board[d]["author"], board[d]["quotes"]) for d in sorted(board))
    return ("board", "## Digest board\n" + (body or "(none yet)"))


def claims_block(run) -> tuple[str, str]:
    by: dict[int, set] = {}
    for c in run.state["ds_claims"]:
        by.setdefault(c["author"], set()).update(c["docs"])
    body = "\n".join(f"Participant {a}: documents {doc_ranges(by[a])}" for a in sorted(by))
    return ("claims", "## Claim board\n" + (body or "(none yet)"))


def analysis_block(run) -> tuple[str, str]:
    """The rebuilt first analysis."""
    a = run.state["ds_analysis"]
    return ("analysis", f"## First analysis ({ANALYSIS_HEAD})\n" + (a["text"] if a else "(none yet)"))


def digest_tool(run, recipients, accepted_note: str) -> ToolSpec:
    def post(ctx: AgentCtx, args: dict) -> ToolResult:
        run_, me = ctx.run, ctx.agent_id
        els = args.get("digests")
        if not isinstance(els, list) or not els:
            ds_refuse(run_, me, "post_digest", "bad_format")
            return ds_result("Refused (bad_format): `digests` must be a non-empty list of {\"doc\": number, \"quotes\": "
                             "[...]} objects. This call used an action.", ok=False,
                             meta={"docs": [], "verdicts": {}, "elements": []})
        check = run_.task.publish_validator(run_.example, run_)
        board = run_.state["ds_board"]
        lines, verdicts, elements, docs = [], {}, [], []
        for el in els:
            doc = el.get("doc") if isinstance(el, dict) else None
            d = as_doc(doc)
            if d is not None and d in board:
                verdict, msg = "duplicate", f"the board already holds a verified digest of document {d}"
            elif not isinstance(el, dict):
                verdict, msg = "bad_format", "each digest must be an object {\"doc\": number, \"quotes\": [...]}"
            else:
                verdict, msg = check(me, doc, el.get("quotes"))
            key = d if d is not None else doc
            docs.append(key)
            elements.append({"doc": key, "verdict": verdict})
            verdicts.setdefault(str(key), verdict)
            if verdict == "ok":
                item = ds_new_item(run_, "digest", f"digest:{d}", me, recipients(run_, me), doc=d,
                                   quotes=[str(q) for q in el.get("quotes") or []])
                board[d] = item
                lines.append(f"Document {d}: verified and added to the digest board{accepted_note}.")
            else:
                ds_refuse(run_, me, "post_digest", verdict, doc=key)
                lines.append(f"Document {key}: refused ({verdict}): {msg}.")
        dup_only = all(e["verdict"] == "duplicate" for e in elements)
        if dup_only:
            lines.append("This call used no action.")
        return ds_result("\n".join(lines), ok=all(e["verdict"] == "ok" for e in elements), free=dup_only,
                         meta={"docs": docs, "verdicts": verdicts, "elements": elements})

    params = {"type": "object", "properties": {"digests": {
        "type": "array", "description": "One entry per document.",
        "items": {"type": "object", "properties": {
            "doc": {**INT, "description": "The document's number (one you have opened)."},
            "quotes": {**STR_LIST, "description": "Quotes copied word for word from that document."}},
            "required": ["doc", "quotes"]}}}, "required": ["digests"]}
    return ds_tool("post_digest", "Post verified digests of documents you have opened (one action per call; free when "
                   "every digest is a duplicate). " + DIGEST_RULES, params, post, on_compact="effect")


def agent_cfg(run, **overrides):
    make = getattr(run, "agent_config", None)
    if make is not None:
        return make(**overrides)
    cfg = run.cfg.agent_config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class DocSweepBoard(BlackboardProtocol):
    name = "docsweep_board"
    note_cap = 6000

    def __init__(self, planning_steps: int = 0, plan_tools=("open_doc", "search"), wait_until: int = 0,
                 claims: bool = True, prompt_file: str = ""):
        super().__init__(wait_until=wait_until, planning_steps=planning_steps, prompt_file=prompt_file)
        self.planning_steps, self.wait_until = int(planning_steps), int(wait_until)
        self.plan_tools = tuple(plan_tools)
        self.claims = bool(claims)

    def build(self, run) -> None:
        """Initialise the delivery state, then build the members as Protocol.build does."""
        n = run.cfg.n_agents + run.cfg.n_failing
        ds_init(run, n)
        run.state["ds_coverage"] = {i: {"covered_step": None, "submit_step": None, "covered_at_submit": None}
                                    for i in range(n)}
        Protocol.build(self, run)

    def blocks(self) -> tuple:
        """The compacted-history blocks of a member: GROUP_BLOCKS, without "claims" when claims are off."""
        return tuple(b for b in GROUP_BLOCKS if self.claims or b != "claims")

    def protocol_text(self, run, agent_id: int) -> str:
        n = run.cfg.n_agents + run.cfg.n_failing
        what = ("each new claim, each newly verified digest and the first analysis is" if self.claims else
                "each newly verified digest and the first analysis is")
        text = DS_PROTOCOL.format(me=agent_id, n=n, last=n - 1, what=what,
                                  claim=DS_CLAIM_DOC.format(c=MAX_CLAIMS, m=MAX_CLAIM_IDS) if self.claims else "")
        if getattr(run.task, "access", "all") == "tree":
            text = text.replace(DS_ALL_ACCESS, DS_SEAT_ACCESS)
        if self.wait_until:
            text += DS_WAIT_NOTE.format(k=self.wait_until + 1)
        if self.planning_steps:
            text += DS_PLAN_NOTE.format(k=self.planning_steps + 1, claim=DS_PLAN_CLAIM if self.claims else "")
        return text

    def method_prompt(self, run, agent_id: int) -> str:
        return ""

    def build_system(self, run, agent_id: int, task_system: str) -> str | None:
        env = run.task.system_prompt(run.example, kind="member", blocks=self.blocks(),
                                     actions=run.cfg.max_actions, steps=run.cfg.max_steps)
        return env + "\n\n" + self.protocol_text(run, agent_id) + self.notes_for(agent_id)

    def extra_tools(self, run, agent_id: int) -> list[ToolSpec]:
        out = [self.claim_tool(run)] if self.claims else []
        n = run.cfg.n_agents + run.cfg.n_failing
        out.append(digest_tool(run, lambda r, me: range(n), "; the other participants receive it at their next call"))
        out.append(self.analysis_tool(run))
        return out

    def submit_tool(self, run, agent_id: int) -> ToolSpec:
        return run.task.submit_tool(run.example, with_refs=False)

    def claim_tool(self, run) -> ToolSpec:
        n = run.cfg.n_agents + run.cfg.n_failing

        def claim(ctx: AgentCtx, args: dict) -> ToolResult:
            run_, me, k = ctx.run, ctx.agent_id, n_docs(ctx.run)
            raw = args.get("doc_ids")
            docs = [as_doc(x) for x in raw] if isinstance(raw, list) else None
            mine = sum(1 for c in run_.state["ds_claims"] if c["author"] == me)
            if docs is None or any(d is None for d in docs):
                verdict, msg = "bad_format", "`doc_ids` must be a list of document numbers"
            elif mine >= MAX_CLAIMS:
                verdict, msg = "claim_limit", f"you have made your {MAX_CLAIMS} claims"
            elif not docs:
                verdict, msg = "empty", "name at least one document"
            elif len(docs) > MAX_CLAIM_IDS:
                verdict, msg = "too_many_ids", f"a claim names at most {MAX_CLAIM_IDS} documents"
            elif len(set(docs)) != len(docs):
                verdict, msg = "repeated_ids", "name each document once"
            elif any(not 1 <= d <= k for d in docs):
                verdict, msg = "invalid_doc", f"documents are numbered 1 to {k}"
            else:
                verdict, msg = "ok", ""
            if verdict != "ok":
                ds_refuse(run_, me, "claim", verdict)
                return ds_result(f"Refused ({verdict}): {msg}. Claims are free: this used no action.", ok=False, free=True,
                                 meta={"docs": [d for d in docs or [] if d is not None], "verdict": verdict})
            item = ds_new_item(run_, "claim", f"claim:{len(run_.state['ds_claims'])}", me, range(n), docs=docs)
            run_.state["ds_claims"].append({"author": me, "step": item["step"], "seq": item["seq"], "docs": docs})
            left = MAX_CLAIMS - mine - 1
            return ds_result(f"Claimed documents {', '.join(str(d) for d in docs)}. The other participants receive this "
                             f"claim at their next call. You can make {left} more claim(s).", free=True,
                             meta={"docs": docs, "verdict": "ok"})

        return ds_tool("claim", f"Tell the other participants which documents you will read. Free: it uses no action. At "
                       f"most {MAX_CLAIMS} claims per participant, each of 1 to {MAX_CLAIM_IDS} distinct document numbers.",
                       {"type": "object", "properties": {"doc_ids": {
                           "type": "array", "items": INT, "description": "The numbers of the documents you will read."}},
                        "required": ["doc_ids"]}, claim, free=True, on_compact="effect")

    def analysis_tool(self, run) -> ToolSpec:
        n = run.cfg.n_agents + run.cfg.n_failing

        def post(ctx: AgentCtx, args: dict) -> ToolResult:
            run_, me = ctx.run, ctx.agent_id
            text = args.get("text")
            text = "" if text is None else str(text)
            if run_.state["ds_analysis"] is not None:
                ds_refuse(run_, me, "post_analysis", "analysis_after_first")
                return ds_result("Refused (analysis_after_first): the group's first analysis is already posted, and only "
                                 "the first is accepted. This used no action.", ok=False, free=True,
                                 meta={"verdict": "analysis_after_first"})
            err = run_.task.analysis_error(run_.example, run_, text)
            if err:
                verdict, msg = err
                free = verdict == "too_long"
                ds_refuse(run_, me, "post_analysis", verdict)
                return ds_result(f"Refused ({verdict}): {msg}" + (" This used no action." if free else ""), ok=False,
                                 free=free, meta={"verdict": verdict})
            tok = getattr(run_.task, "chat_tokenizer", None)
            tokens = tok().count(text) if tok is not None else None
            item = ds_new_item(run_, "analysis", "analysis", me, range(n), text=text, tokens=tokens)
            run_.state["ds_analysis"] = item
            return ds_result(f"Analysis posted. Every other participant receives it at its next call, under the heading "
                             f"\"{ANALYSIS_HEAD}\".", meta={"verdict": "ok"})

        return ds_tool("post_analysis", "Post an analysis or recommendation in free text for every other participant (one "
                       "action; at most 1,500 tokens, a longer one refused at no cost). Only a participant that has covered "
                       "every document can post one, and only the group's first analysis is accepted.",
                       {"type": "object", "properties": {"text": {**STR, "description": "Your analysis or recommendation."}},
                        "required": ["text"]}, post, on_compact="effect")

    def held(self, run, agent) -> bool:
        return bool(self.wait_until and run.step < self.wait_until and agent.id not in run.state["ds_analysis_seen"]
                    and not run.task.holds_every_fact(run.example, run, agent.id))

    def gate(self, run, agent, tool_name: str) -> str | None:
        if self.planning_steps and tool_name in self.plan_tools and run.step < self.planning_steps:
            ds_hold(run, agent, "plan", tool_name)
            return Protocol.Hold(DS_PLAN_HOLD.format(k=self.planning_steps + 1, claim=DS_PLAN_HOLD_CLAIM if self.claims else ""))
        if tool_name == "post_analysis" and not run.task.holds_every_fact(run.example, run, agent.id):
            ds_refuse(run, agent.id, "post_analysis", "not_covered")
            return DS_COVER_REFUSED.format(c=covered_count(run, agent.id), k=n_docs(run))
        if tool_name == agent.submit_name and self.held(run, agent):
            ds_hold(run, agent, "timing", tool_name)
            return Protocol.Hold(DS_HOLD.format(k=self.wait_until + 1))
        return None

    def defer_final(self, run, agent) -> bool:
        """Out of actions while held by the timing rule: frozen until the hold lifts (kind "frozen", logged once)."""
        if not self.held(run, agent):
            return False
        frozen = run.state.setdefault("frozen", {})
        if agent.id not in frozen:
            frozen[agent.id] = run.step
            ds_hold(run, agent, "frozen", None)
        return True

    def injection(self, run, agent, step: int) -> str | None:
        return ds_offer(run, agent)

    def rebuilt_items(self, run, agent) -> list[str]:
        """The items in this member's rebuilt blocks: every digest, every claim (claims on) and the analysis."""
        out = [it["id"] for it in run.state["ds_board"].values()]
        if self.claims:
            out += [it["id"] for it in run.state["ds_items"] if it["kind"] == "claim"]
        if run.state["ds_analysis"] is not None:
            out.append("analysis")
        return out

    def on_prompt(self, run, agent, call: int, via: str) -> None:
        ds_confirm(run, agent, call, via, self.rebuilt_items(run, agent) if via == "rebuilt" else [])

    def memory_blocks(self, run, agent) -> list[tuple[str, str]]:
        """board, claims (when claims are on) and analysis, each always present ("(none yet)" when empty)."""
        out = [board_block(run)]
        if self.claims:
            out.append(claims_block(run))
        out.append(analysis_block(run))
        return out

    async def before_step(self, run, step: int) -> None:
        return None

    def live_block(self, run, agent, step: int) -> str | None:
        return None

    async def after_step(self, run, step: int) -> None:
        """Coverage bookkeeping: per member, the first step at which task.holds_every_fact holds (covered_step)."""
        cov = run.state["ds_coverage"]
        for a in run.agents:
            c = cov.setdefault(a.id, {"covered_step": None, "submit_step": None, "covered_at_submit": None})
            if c["covered_step"] is None and run.task.holds_every_fact(run.example, run, a.id):
                c["covered_step"] = step

    def on_agent_done(self, run, agent) -> None:
        """Coverage bookkeeping: the submission step and how many documents the member had covered then."""
        c = run.state["ds_coverage"].setdefault(agent.id, {"covered_step": None, "submit_step": None,
                                                           "covered_at_submit": None})
        sub = ((agent.ctx.state.get("ds") or {}).get("submitted_step"))
        c["submit_step"] = sub if sub is not None else run.step
        c["covered_at_submit"] = covered_count(run, agent.id)

    def collect(self, run) -> list[dict]:
        return Protocol.collect(self, run)

    def dump(self, run) -> dict:
        """The protocol settings, items, board, claims, analysis, refusals, holds and coverage."""
        st = run.state
        sha = None
        if self.prompt_file:
            from pathlib import Path
            sha = hashlib.sha256(Path(self.prompt_file).read_bytes()).hexdigest()
        a = st["ds_analysis"]
        return {"protocol": {"planning_steps": self.planning_steps, "plan_tools": list(self.plan_tools),
                             "wait_until": self.wait_until, "claims": self.claims, "prompt_file": self.prompt_file,
                             "prompt_sha256": sha},
                "items": st["ds_items"],
                "board": {d: st["ds_board"][d] for d in sorted(st["ds_board"])},
                "claims": st["ds_claims"],
                "analysis": ({k: a[k] for k in ("author", "step", "seq", "text", "tokens")} if a else None),
                "refusals": st["ds_refusals"],
                "holds": st.get("holds", []),
                "coverage": {i: st["ds_coverage"][i] for i in sorted(st["ds_coverage"])},
                "prompts": self.prompt_check(run)}

    def prompt_check(self, run) -> dict:
        names = (("claim",) if self.claims else ()) + ("post_digest", "post_analysis")
        out = {}
        for a in run.agents:
            sysp = a.system_prompt or ""
            out[a.id] = {"system_sha256": hashlib.sha256(sysp.encode("utf-8")).hexdigest(), "tools": sorted(a.tools),
                         "describes": {n: (f"`{n}`" in sysp and n in a.tools) for n in names}}
            if self.prompt_file:
                block = self.notes_for(a.id).strip()
                out[a.id]["notes_in_prompt"] = bool(block) and block in sysp
        return out
