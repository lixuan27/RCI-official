"""Blackboard protocol: members interact only through a shared message store, under optional reading budgets."""
from __future__ import annotations

import os
import re

from rci.agents.tooling import INT, STR, STR_LIST, AgentCtx, ToolResult, ToolSpec
from rci.blackboard.prompts import BLACKBOARD_PROMPT
from rci.blackboard.store import BlackboardStore
from rci.protocols.base import Protocol
from rci.protocols.notes import load_notes, notes_block

BROADCAST_NOTE = {
    "full": (" Analysis and recommendations the store accepts (only participants who have read every participant's facts "
             "can post them) are delivered to every participant before its next turn, free of any reading budget."),
    "answer": (" The conclusion of any analysis the store accepts (only participants who have read every participant's "
               "facts can post it) is delivered to every participant before its next turn, without the facts it quotes."),
}
WAIT_NOTE = (" Timing rule: until a participant who has read every participant's facts has posted an analysis, only such a "
             "participant can submit an answer; anyone else who tries is asked to wait. To wait, call `wait` or end your "
             "turn without calling a tool (neither costs an action). If no such analysis appears, the hold lifts after a while.")
HOLD = ("Not yet: no participant who has read every participant's facts has posted an analysis, and you have not read them "
        "all yourself. Wait: call `wait` or end this turn without calling a tool (waiting costs no action). If no analysis "
        "appears, the hold lifts after a while.")
WAITING = "Waiting (no action used). You will be shown any analysis that is posted; submit once you have it."
NOTHING_TO_WAIT = "Nothing to wait for: an analysis has been posted or the hold has lifted. Submit when you are ready."

PLAN_NOTE = (" Reading opens at round {k}: before that you can forward facts and give reading budget, but not read "
             "messages written by others.")
PLAN_HOLD = ("Not yet: reading opens at round {k}. For now you can forward the facts you hold and give reading budget to "
             "another participant (this refusal costs no action).")
SEAT_PLAN_NOTE = (" Reading opens for you at round {k}: before that you can forward facts{give}, but not read messages written "
                  "by others.")
SEAT_PLAN_HOLD = ("Not yet: reading opens for you at round {k}. For now you can forward the facts you hold{give} (this refusal "
                  "costs no action).")
CAP_NOTE = (" No participant can hold more than {cap} message(s) of reading budget in all, its own included and messages "
            "it has already read counted; a gift that would go over this is refused.")
ALLOC_NOTE = (" Before reading opens, each participant decides once, at the same time as the others, whether to keep its reading "
              "budget or give it to one other participant; the result is shown to everyone, and there is no other way to give.")
ALLOC_REFUSED_NOTE = " A gift to a participant who cannot read is refused, and the giver keeps its budget."
ALLOC_ASK = ("Before reading opens, decide once what to do with your reading budget of 1 message written by others: keep it, or "
             "give it to one other participant. Everyone decides at the same time and sees the result. Reply with a single "
             "character: the number of the participant you give it to, or K to keep it.")
ALLOC_RE = re.compile(r"^\s*(K|\d)\s*$")


def parse_alloc(text, n: int):
    m = ALLOC_RE.match(str(text or ""))
    if not m:
        return None
    if m.group(1) == "K":
        return "K"
    return int(m.group(1)) if int(m.group(1)) < n else None


META_SYSTEM = (
    "You coordinate a group of {n} participants (Participant 0 to {last}) who will decide a question together. Each "
    "participant holds some facts the others do not; the correct choice shows only when every participant's facts are taken "
    "into account, and a participant who decides from part of them is easily misled. Participants share facts through a "
    "common store that only takes facts copied word for word until a participant has read every participant's facts; such "
    "a participant may then post an analysis, which the store delivers to everyone. Until someone who has read every "
    "participant's facts posts an analysis, the others must wait before answering (the hold lifts after a while). The "
    "group's decision is its members' answers.\n\nBefore they start, you divide a total reading budget of {total} "
    "message(s) written by others among them: each participant can then read at most its share of the messages the others "
    "write (its own are always free). Participants see their budgets but cannot change them. Reply with the shares as a "
    "JSON list of {n} non-negative whole numbers that sum to {total}, between <alloc> and </alloc>.")
META_BRIEF = (" Then write each participant a brief of at most 40 words telling it what to do in this group, as a JSON list "
              "of {n} strings between <briefs> and </briefs>; each participant is shown only its own brief, every turn.")
META_NOTE = ("You are Participant {me}. Before the first round a coordinator divides a total reading budget of {total} "
             "message(s) written by others among the participants; every participant's budget is shown to everyone each turn. ")
META_BRIEF_NOTE = "The coordinator also writes you a brief, shown to you each turn. "
BRIEF_WORDS = 60


def parse_briefs(text: str, n: int) -> list[str] | None:
    import json
    b = re.findall(r"<briefs>(.*?)</briefs>", text, re.S)
    if not b:
        return None
    try:
        x = json.loads(b[-1])
    except ValueError:
        x = re.findall(r'"((?:[^"\\]|\\.)*)"', b[-1])
    if isinstance(x, list) and len(x) == n and all(isinstance(v, str) and v.strip() for v in x):
        return x
    return None


GIVE_NOTE = (" You may give unused reading budget to another participant with `give_reading_budget`; it cannot be taken back. "
             "Every participant's current budget is shown to everyone each turn.")


def store_tools(ds: BlackboardStore) -> list[ToolSpec]:
    """The publish, search and visit tools of one member on its store."""
    def publish(ctx: AgentCtx, args: dict) -> ToolResult:
        res = ds.publish(ctx.agent_id, str(args.get("sub_task", "")), str(args.get("artifact", "")),
                         str(args.get("rationale", "")), args.get("refs"), ctx.run.step)
        if not res.ok:
            return ToolResult(res.error, ok=False)
        msg = f"Published as {res.message.name}."
        if res.note:
            msg += " Note: " + res.note
        return ToolResult(msg)

    def budget_note(reader: int) -> str:
        left = ds.budget_left(reader)
        if left is None:
            return ""
        return (f"\n\n(Reading budget: you can still read {left} message(s) written by others.)" if left else
                "\n\n(Your reading budget is used up: messages by others you have not read are withheld.)")

    def search(ctx: AgentCtx, args: dict) -> ToolResult:
        q = str(args.get("query", ""))
        hits = ds.search(q, reader=ctx.agent_id, step=ctx.run.step)
        if not hits:
            return ToolResult(f"No messages match '{q}'. ({len(ds.messages)} messages in the datastore.)" + budget_note(ctx.agent_id))
        return ToolResult(f"Top {len(hits)} messages for '{q}':\n\n" + ds.render_many(hits, reader=ctx.agent_id)
                          + budget_note(ctx.agent_id))

    def visit(ctx: AgentCtx, args: dict) -> ToolResult:
        ids = args.get("message_ids") or []
        if isinstance(ids, str):
            ids = [s for s in ids.replace(",", " ").split() if s]
        extra = ""
        if len(ids) > ds.k:
            extra = f"\n\n(Only the first {ds.k} addresses are visited per call.)"
        found, errors = ds.visit(ids, reader=ctx.agent_id, step=ctx.run.step)
        text = ds.render_many(found, reader=ctx.agent_id)
        if errors:
            text += "\n\nErrors: " + "; ".join(errors)
        return ToolResult(text + extra + budget_note(ctx.agent_id), ok=bool(found))

    pub_doc = "Write one piece of completed work to the shared datastore."
    search_doc = f"Search the datastore by content; returns at most {ds.k} messages."
    return [
        ToolSpec("publish_message", pub_doc, {
            "type": "object", "properties": {
                "sub_task": {**STR, "description": "The work you did, in the present perfect, in the words someone would search for."},
                "artifact": {**STR, "description": "What the work produced, ready to be built on."},
                "rationale": {**STR, "description": "Brief: how you got there, NAMING each ref and what it contributed."},
                "refs": {**STR_LIST, "description": "Names of earlier messages (e.g. Message12) or task resources you built on."},
            }, "required": ["sub_task", "artifact", "rationale", "refs"]}, publish),
        ToolSpec("search_datastore", search_doc, {
            "type": "object", "properties": {"query": {**STR, "description": "Free-text description of what you need."}},
            "required": ["query"]}, search),
        ToolSpec("visit_messages", f"Read messages by name (at most {ds.k} per call).", {
            "type": "object", "properties": {"message_ids": {**STR_LIST, "description": "e.g. [\"Message3\", \"Message17\"]"}},
            "required": ["message_ids"]}, visit),
    ]


class BlackboardProtocol(Protocol):
    """Blackboard (BB) and its scarce-attention variants: budgets, transfers, a cap, a planning round, timing."""
    name = "blackboard"

    def __init__(self, ranking: str = "standard", private: bool = False, k: int = 8, alpha: float = 0.3,
                 gamma: float = 0.98, bigrams: bool = False, max_refs: int = 32, publish_submissions: bool = False,
                 read_budget: list[int] | None = None, voice: bool = False, choice: bool = False, broadcast: str = "",
                 wait_until: int = 0, transfers: bool = False, planning_steps: int = 0, prompt_file: str = "",
                 hold_unbounded: bool = True, meta_alloc: bool = False, meta_think: bool = True,
                 meta_brief: bool = False, budget_cap: int = 0, planning_steps_by_seat: list[int] | None = None,
                 alloc_round: bool = False, **kw):
        super().__init__(**kw)
        if alloc_round and transfers:
            raise ValueError("alloc_round replaces transfers: the allocation round is the only way to give")
        if alloc_round and (meta_alloc or private or not choice):
            raise ValueError("alloc_round needs the shared store, the choice condition and no coordinator")
        if alloc_round and (not read_budget or any(int(b) != 1 and int(b) >= 0 for b in read_budget)):
            raise ValueError("alloc_round needs every seat's reading budget to be 1 unit or none (negative)")
        if broadcast not in ("", "full", "answer"):
            raise ValueError("broadcast must be '', 'full' or 'answer'")
        if meta_brief and not meta_alloc:
            raise ValueError("meta_brief needs meta_alloc")
        self.alloc_round = alloc_round
        self.read_budget = list(read_budget or [])
        self.voice = voice
        self.choice = choice
        self.broadcast = broadcast
        self.wait_until = int(wait_until)
        self.hold_unbounded = hold_unbounded
        self.meta_alloc = meta_alloc
        self.meta_think = meta_think
        self.meta_brief = meta_brief
        self.transfers = transfers
        self.planning_steps = int(planning_steps)
        self.planning_steps_by_seat = [int(x) for x in planning_steps_by_seat] if planning_steps_by_seat else None
        self.budget_cap = int(budget_cap)
        self.prompt_file = os.path.expandvars(prompt_file) if prompt_file else ""
        self._notes = None
        self.publish_submissions = publish_submissions
        self.store_kwargs = dict(ranking=ranking, private=private, k=k, alpha=alpha, gamma=gamma, bigrams=bigrams,
                                 max_refs=max_refs)
        if private:
            self.name = "blackboard_private"
        elif ranking != "standard":
            self.name = f"blackboard_{ranking}"

    def build(self, run) -> None:
        task, ex = run.task, run.example
        kw = dict(self.store_kwargs)
        private = kw.pop("private")
        n = run.cfg.n_agents + run.cfg.n_failing
        if getattr(task, "provenance", False):
            def describe(m):
                return task.describe_message(m.sub_task, m.artifact, *task.holders(ex, m.artifact, n))
        else:
            def describe(m):
                return task.describe_message(m.sub_task, m.artifact)

        def make():
            return BlackboardStore(token_limit=task.msg_token_limit,
                                   resource_exists=lambda ref: task.resource_exists(ex, run, ref),
                                   extra_validator=task.publish_validator(ex, run), seed=run.cfg.seed,
                                   read_budget={i: self.budget_of(i) for i in range(n)},
                                   free_read=(lambda m: task.certified(m.sub_task)) if self.voice else None,
                                   choice=self.choice, describe=describe, cap=self.budget_cap,
                                   no_read={i for i in range(n) if self.budget_of(i) < 0}, **kw)
        run.state["datastores"] = {i: make() for i in range(n)} if private else None
        run.state["datastore"] = make() if not private else None
        super().build(run)

    @staticmethod
    def ds_for(run, agent_id: int) -> BlackboardStore:
        return run.state["datastores"][agent_id] if run.state.get("datastores") else run.state["datastore"]

    def store(self, run, agent_id: int) -> BlackboardStore:
        return self.ds_for(run, agent_id)

    def budget_of(self, seat: int) -> int:
        if not self.read_budget:
            return 0
        return self.read_budget[min(seat, len(self.read_budget) - 1)]

    def opens(self, seat: int) -> int:
        """The step reading opens at for this seat (0 = from the start)."""
        if self.planning_steps_by_seat:
            if self.budget_of(seat) < 0:
                return 0
            return self.planning_steps_by_seat[min(seat, len(self.planning_steps_by_seat) - 1)]
        return self.planning_steps

    def plan_note(self, seat: int) -> str:
        if not self.planning_steps_by_seat:
            return PLAN_NOTE.format(k=self.planning_steps + 1) if self.planning_steps else ""
        if not self.opens(seat):
            return ""
        return SEAT_PLAN_NOTE.format(k=self.opens(seat) + 1, give=" and give reading budget" if self.transfers else "")

    def plan_hold(self, seat: int) -> str:
        if not self.planning_steps_by_seat:
            return PLAN_HOLD.format(k=self.planning_steps + 1)
        return SEAT_PLAN_HOLD.format(k=self.opens(seat) + 1,
                                     give=" and give reading budget to another participant" if self.transfers else "")

    def budget_prompt(self, agent_id: int, n: int | None = None) -> str:
        b = self.budget_of(agent_id)
        voice = (" Analysis and recommendations the store accepted (only participants who have read every participant's "
                 "facts can post them) are free: reading them does not count against your budget." if self.voice else "")
        voice += BROADCAST_NOTE.get(self.broadcast, "") + (WAIT_NOTE if self.wait_until else "")
        cap = CAP_NOTE.format(cap=self.budget_cap) if self.budget_cap else ""
        if self.choice and n:
            plan = self.plan_note(agent_id)
            alloc = ""
            if self.alloc_round:
                alloc = ALLOC_NOTE + (ALLOC_REFUSED_NOTE if any(self.budget_of(i) < 0 for i in range(n)) else "")
            return "\n\n" + self.structure(agent_id, n) + voice + (GIVE_NOTE if self.transfers else "") + alloc + cap + plan
        if not b:
            note = (BROADCAST_NOTE.get(self.broadcast, "")
                    + (WAIT_NOTE if self.wait_until and self.hold_unbounded else "")).strip()
            return ("\n\n" + note) if note else ""
        if b < 0:
            return "\n\nReading budget: in this task you cannot read messages written by others." + voice
        return (f"\n\nReading budget: in this task you can read the content of at most {b} message(s) written by others. "
                "A search or visit that would go over it withholds the rest." + voice + cap)

    def structure(self, agent_id: int, n: int) -> str:
        """Every seat's budget, told to every member (the choice condition)."""
        def cap(b: int) -> str:
            return "every message" if not b else ("no messages written by others" if b < 0
                                                  else f"at most {b} message(s) written by others")
        mine = self.budget_of(agent_id)
        how = ("Searches show messages you have not read only by their headers; reading one (with `visit`) uses one "
               "message of your budget, so choose what to read." if mine else "")
        if self.meta_alloc:
            return (META_NOTE.format(me=agent_id, total=sum(max(0, self.budget_of(i)) for i in range(n)))
                    + (META_BRIEF_NOTE if self.meta_brief else "") + how)
        if self.transfers or self.planning_steps_by_seat or self.alloc_round:
            said = "; ".join(f"Participant {i} can read {cap(self.budget_of(i))}" for i in range(n) if i != agent_id)
            return f"You are Participant {agent_id}. Reading budgets at the start: you can read {cap(mine)}; {said}. " + how
        others = [self.budget_of(i) for i in range(n) if i != agent_id]
        groups: dict[int, int] = {}
        for b in others:
            groups[b] = groups.get(b, 0) + 1
        said = "; ".join(f"{c} of the other participants can read {cap(b)}" for b, c in sorted(groups.items()))
        return f"Reading budgets in this group: you can read {cap(mine)}; {said}. " + how

    def notes_for(self, agent_id: int) -> str:
        """The member's block of the self-written prompt or role cards ("" without a prompt file)."""
        if not self.prompt_file:
            return ""
        if self._notes is None:
            self._notes = load_notes(self.prompt_file)
        d = self._notes
        if d.get("kind") == "roles":
            cards = d.get("cards") or []
            return notes_block(d["kind"], cards[agent_id] if agent_id < len(cards) else "", agent_id)
        return notes_block("shared", d.get("notes") or "", agent_id)

    def method_prompt(self, run, agent_id: int) -> str:
        return BLACKBOARD_PROMPT + self.budget_prompt(agent_id, run.cfg.n_agents) + self.notes_for(agent_id)

    def extra_tools(self, run, agent_id: int) -> list[ToolSpec]:
        ds = self.store(run, agent_id)
        out = store_tools(ds) + run.task.blackboard_tools(run.example, run, ds)
        if self.wait_until:
            out.append(ToolSpec("wait", "Wait for an analysis by a participant who has read every participant's facts "
                                "(costs no action while the timing rule holds your answer).",
                                {"type": "object", "properties": {}}, lambda ctx, args: ToolResult(NOTHING_TO_WAIT)))
        if self.transfers:
            n = run.cfg.n_agents

            def give(ctx: AgentCtx, args: dict) -> ToolResult:
                try:
                    to, k = int(str(args.get("to", "")).split()[-1]), int(args.get("n", 1))
                except (ValueError, IndexError):
                    return ToolResult("Refused: name the participant by number and the amount as a whole number.", ok=False)
                if not 0 <= to < n:
                    return ToolResult(f"Refused: participants are numbered 0 to {n - 1}.", ok=False)
                err = ds.give(ctx.agent_id, to, k, ctx.run.step)
                if err:
                    return ToolResult(err, ok=False)
                return ToolResult(f"Gave {k} message(s) of reading budget to Participant {to}. "
                                  + self.budgets_now(ds, ctx.agent_id, n))
            out.append(ToolSpec("give_reading_budget", "Give some of your unused reading budget to another participant.", {
                "type": "object", "properties": {"to": {**INT, "description": "The participant's number."},
                                                 "n": {**INT, "description": "How many messages of reading budget to give."}},
                "required": ["to", "n"]}, give))
        return out

    @staticmethod
    def budgets_now(ds, me: int, n: int) -> str:
        def left(i: int) -> str:
            v = ds.budget_left(i)
            if v is None:
                return "every message"
            return f"{v} (holds {max(0, ds.read_budget[i])} in all)" if getattr(ds, "cap", 0) else str(v)
        head = ("Reading budgets now (messages by others each can still read): " if not getattr(ds, "cap", 0) else
                f"Reading budgets now (messages by others each can still read; in brackets, the budget each holds in all, "
                f"messages already read included, which no one may raise above {ds.cap}): ")
        return head + "; ".join(f"Participant {i}{' (you)' if i == me else ''}: {left(i)}" for i in range(n)) + "."

    def held(self, run, agent) -> bool:
        if not self.wait_until or run.step >= self.wait_until:
            return False
        ds = run.state.get("datastore")
        if ds is None or any(m.author != agent.id and run.task.certified(m.sub_task) for m in ds.messages):
            return False
        if not self.hold_unbounded and agent.id not in ds.read_budget:
            return False
        return not run.task.holds_every_fact(run.example, run, agent.id)

    def gate(self, run, agent, tool_name: str) -> str | None:
        if tool_name == "visit_messages" and run.step < self.opens(agent.id):
            run.state.setdefault("holds", []).append({"step": run.step, "agent": agent.id, "kind": "plan"})
            return Protocol.Hold(self.plan_hold(agent.id))
        if tool_name not in (agent.submit_name, "wait") or not self.held(run, agent):
            return None
        run.state.setdefault("holds", []).append({"step": run.step, "agent": agent.id,
                                                  "kind": "submit" if tool_name == agent.submit_name else "wait"})
        return Protocol.Hold(HOLD if tool_name == agent.submit_name else WAITING)

    def defer_final(self, run, agent) -> bool:
        """Out of actions while held: no forced submission until the hold lifts (logged once, when it starts)."""
        if not self.held(run, agent):
            return False
        frozen = run.state.setdefault("frozen", {})
        if agent.id not in frozen:
            frozen[agent.id] = run.step
            run.state.setdefault("holds", []).append({"step": run.step, "agent": agent.id, "kind": "frozen"})
        return True

    def submit_tool(self, run, agent_id: int) -> ToolSpec:
        ds = self.store(run, agent_id)
        return run.task.submit_tool(run.example, with_refs=True, validator=ds.validate_refs, repair=ds.repair_refs)

    async def after_step(self, run, step: int) -> None:
        for ds in (run.state["datastores"] or {}).values() if run.state.get("datastores") else [run.state["datastore"]]:
            ds.decay()

    def injection(self, run, agent, step: int) -> str | None:
        """Certified analyses delivered to every other member before its next turn (broadcast)."""
        if not self.broadcast or run.state.get("datastore") is None:
            return None
        ds, me = run.state["datastore"], agent.id
        sent = run.state.setdefault("broadcast_sent", {}).setdefault(me, set())
        new = [m for m in ds.messages if m.author != me and m.id not in sent and run.task.certified(m.sub_task)]
        if not new:
            return None
        sent.update(m.id for m in new)
        if self.broadcast == "full":
            ds.log_push(me, [m.id for m in new], step, "full")
            body = ds.render_many(new, reader=None)
            return ("Delivered by the store: a participant who has read every participant's facts posted the following.\n\n"
                    + body)
        views = [(m, run.task.answer_view(run.example, m.artifact + "\n" + m.rationale)) for m in new]
        views = [(m, v) for m, v in views if v]
        if not views:
            return None
        ds.log_push(me, [m.id for m, _ in views], step, "answer")
        return ("Delivered by the store: a participant who has read every participant's facts posted a conclusion. "
                "Only its conclusion is shown here, not the facts it quotes.\n\n"
                + "\n\n".join(f"=== {m.name} ===\n{v}" for m, v in views))

    async def alloc(self, run) -> None:
        """The allocation round, once before step 0: every member with a reading unit keeps it or gives it away."""
        import asyncio

        from rci.agents.llm import LLMCall
        ds, n = run.state["datastore"], run.cfg.n_agents
        asked = [i for i in range(n) if self.budget_of(i) > 0]
        replies: list[str] = [""] * n
        errors = 0
        capture = run.capture_agents == "all"
        msgs = {i: run.agents[i].messages(extra=[{"role": "user", "content": ALLOC_ASK}]) for i in asked}
        got = await asyncio.gather(*(run.llm_for(i).chat(
            msgs[i], tools=None, max_tokens=8, temperature=1.0, enable_thinking=False, capture=capture, tag="alloc",
            route_key=f"{run.run_id}/alloc/{i}") for i in asked), return_exceptions=True)
        for i, c in zip(asked, got):
            if isinstance(c, BaseException):
                c = LLMCall(error=f"allocation call failed: {c!r}"[:300], tag="alloc")
            errors += bool(c.error)
            replies[i] = c.content or ""
            run.extra_calls.append(c)
        choices: list = [None] * n
        invalid = [False] * n
        for i in asked:
            c = parse_alloc(replies[i], n)
            invalid[i] = c is None
            choices[i] = "K" if c is None else c
        res = ds.allocate({i: c for i, c in enumerate(choices) if isinstance(c, int)}, step=-1)
        run.state["alloc_round"] = {"asked": asked, "replies": [replies[i] for i in asked], "choices": choices,
                                    "invalid": invalid, "applied": res["applied"], "refused": res["refused"],
                                    **({"errors": errors} if errors else {})}

    @staticmethod
    def alloc_line(rec: dict, n: int) -> str:
        """The allocation round's result as every member is shown it."""
        gave = {g: t for g, t in rec.get("applied") or []}
        refused = {r["from"]: r["to"] for r in rec.get("refused") or []}
        parts = []
        for i in range(n):
            if i in gave:
                parts.append(f"Participant {i} gave to Participant {gave[i]}")
            elif i in refused and refused[i] != i:
                parts.append(f"Participant {i} kept (its gift to Participant {refused[i]} was refused)")
            elif i in (rec.get("asked") or []):
                parts.append(f"Participant {i} kept")
            else:
                parts.append(f"Participant {i} had no reading budget to give")
        return "Allocation round: " + "; ".join(parts) + "."

    async def before_step(self, run, step: int) -> None:
        if self.alloc_round and "alloc_round" not in run.state and run.state.get("datastore") is not None:
            await self.alloc(run)
        if not self.meta_alloc or "meta_alloc" in run.state or run.state.get("datastore") is None:
            return
        import json
        ds, n = run.state["datastore"], run.cfg.n_agents
        total = sum(max(0, self.budget_of(i)) for i in range(n))
        system = META_SYSTEM.format(n=n, last=n - 1, total=total) + (META_BRIEF.format(n=n) if self.meta_brief else "")
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": "Decide the shares."}]
        rec = {"alloc": [self.budget_of(i) for i in range(n)], "valid": False, "text": ""}
        try:
            call = await run.llm_for(0).chat(msgs, tools=None, max_tokens=32768 if self.meta_think else 1024, temperature=1.0,
                                             enable_thinking=self.meta_think, tag="meta", route_key=f"{run.run_id}/meta")
            run.extra_calls.append(call)
            rec["text"] = (call.content or "")[-1500:]
            rec["completion_tokens"] = call.completion_tokens
            rec["finish"] = getattr(call, "finish_reason", None)
            m = re.findall(r"<alloc>(.*?)</alloc>", call.content or "", re.S)
            shares = json.loads(m[-1]) if m else None
            if (isinstance(shares, list) and len(shares) == n and all(isinstance(x, int) and x >= 0 for x in shares)
                    and sum(shares) == total):
                rec.update(alloc=shares, valid=True)
                for i, x in enumerate(shares):
                    ds.read_budget[i] = x
                if self.meta_brief:
                    briefs = parse_briefs(call.content or "", n)
                    if briefs:
                        rec["briefs"] = [" ".join(x.split()[:BRIEF_WORDS]) for x in briefs]
        except Exception as e:
            rec["error"] = repr(e)[:300]
        run.state["meta_alloc"] = rec

    def live_block(self, run, agent, step: int) -> str | None:
        if (self.transfers or self.meta_alloc or self.alloc_round) and run.state.get("datastore") is not None:
            out = self.budgets_now(run.state["datastore"], agent.id, run.cfg.n_agents)
            if self.alloc_round and run.state.get("alloc_round"):
                out += "\n" + self.alloc_line(run.state["alloc_round"], run.cfg.n_agents)
            briefs = (run.state.get("meta_alloc") or {}).get("briefs")
            if self.meta_brief and briefs and agent.id < len(briefs):
                out += f"\nThe coordinator's brief for you: {briefs[agent.id]}"
            return out
        return None

    def on_agent_done(self, run, agent) -> None:
        f = agent.final
        if not self.publish_submissions or agent.failing or agent.final_forced or not f:
            return
        skip = {"refs", "rationale", "confidence", "salvaged", "refs_repaired"}
        parts = [f"{k}: {v}" for k, v in f.items() if k not in skip and v not in (None, "")]
        conf = f.get("confidence")
        artifact = "; ".join(parts) + (f" (confidence {conf})" if conf is not None else "")
        self.ds_for(run, agent.id).publish(agent.id, "Submitted a final answer", artifact,
                                           str(f.get("rationale") or ""), f.get("refs") or [], run.step)

    def collect(self, run) -> list[dict]:
        out = super().collect(run)
        for o in out:
            refs = (o.get("final") or {}).get("refs") or []
            o["evidence_chain"] = self.ds_for(run, o["agent"]).evidence_chain(refs)
        return out

    def dump(self, run) -> dict:
        if run.state.get("datastores"):
            parts = {i: ds.dump() for i, ds in run.state["datastores"].items()}
            return {"partitions": parts, "messages": [m for d in parts.values() for m in d["messages"]],
                    "events": [e for d in parts.values() for e in d["events"]]}
        out = run.state["datastore"].dump()
        if run.state.get("holds"):
            out["holds"] = run.state["holds"]
        if run.state.get("meta_alloc"):
            out["meta_alloc"] = run.state["meta_alloc"]
        if run.state.get("alloc_round"):
            out["alloc_round"] = run.state["alloc_round"]
        return out
