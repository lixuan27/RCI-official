"""The blackboard store: append-only linked messages, reading budgets and the deterministic rules around them."""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

from rci.agents.tokens import count_tokens, truncate_to_tokens
from rci.blackboard.bm25 import IncrementalBM25

MESSAGE_RE = re.compile(r"^Message(\d+)$")
RANKINGS = ("standard", "bm25", "random")


@dataclass
class Message:
    id: int
    sub_task: str
    artifact: str
    rationale: str
    refs: list[str]
    author: int
    step: int
    msg_refs: list[int] = field(default_factory=list)
    cited_by: list[int] = field(default_factory=list)
    weight: float = 0.0
    truncated: bool = False

    @property
    def name(self) -> str:
        return f"Message{self.id}"


@dataclass
class PublishResult:
    ok: bool
    message: Message | None = None
    error: str | None = None
    note: str | None = None


def _names_ref(rationale: str, ref: str) -> bool:
    pat = r"(?<![A-Za-z0-9_])" + re.escape(ref) + r"(?![A-Za-z0-9_])"
    return re.search(pat, rationale, flags=re.IGNORECASE) is not None


def _short(text: str, n_words: int = 24) -> str:
    words = text.split()
    return " ".join(words[:n_words]) + (" ..." if len(words) > n_words else "")


class BlackboardStore:
    """Append-only message store with citation-weighted BM25 search."""

    def __init__(
        self,
        *,
        alpha: float = 0.3,
        gamma: float = 0.98,
        s0: float = 0.0,
        k: int = 8,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        bigrams: bool = False,
        token_limit: int = 256,
        max_refs: int = 32,
        max_citers_shown: int = 8,
        ranking: str = "standard",
        private: bool = False,
        resource_exists: Callable[[str], bool] | None = None,
        extra_validator: Callable[[int, str, str, str, list[str]], str | None] | None = None,
        seed: int = 0,
        read_budget: dict[int, int] | None = None,
        free_read: Callable[["Message"], bool] | None = None,
        choice: bool = False,
        describe: Callable[["Message"], str] | None = None,
        cap: int = 0,
        no_read: set[int] | None = None,
    ):
        if ranking not in RANKINGS:
            raise ValueError(f"ranking must be one of {RANKINGS}")
        self.alpha, self.gamma, self.s0, self.k = alpha, gamma, s0, k
        self.token_limit, self.max_refs = token_limit, max_refs
        self.max_citers_shown = max_citers_shown
        self.ranking, self.private = ranking, private
        self.resource_exists = resource_exists or (lambda ref: False)
        self.extra_validator = extra_validator
        self.messages: list[Message] = []
        self.index = IncrementalBM25(k1=bm25_k1, b=bm25_b, bigrams=bigrams)
        self.rng = random.Random(seed)
        self.events: list[dict] = []
        self.read_budget = {int(r): max(0, int(b)) for r, b in (read_budget or {}).items() if int(b) != 0}
        self.read: dict[int, set[int]] = {}
        self.free_read = free_read
        self.choice = choice
        self.pushed: dict[int, set[int]] = {}
        self.describe = describe or (lambda m: "a message")
        self.cap = int(cap)
        self.no_read = set(no_read or ())

    def budget_left(self, reader: int | None) -> int | None:
        """How many more messages by others `reader` may see; None when unlimited."""
        if reader is None or reader not in self.read_budget:
            return None
        return max(0, self.read_budget[reader] - len(self.read.get(reader, set())))

    def give(self, giver: int, to: int, n: int, step: int, round=None) -> str | None:
        err, op = None, "give_refused"
        extra = {} if round is None else {"round": round}
        if giver not in self.read_budget:
            err = "Refused: your reading is not limited, so you have no budget to give."
        elif to == giver:
            err = "Refused: give to another participant."
        elif to not in self.read_budget:
            err = f"Refused: Participant {to} can already read every message."
        elif not 1 <= n <= (self.budget_left(giver) or 0):
            err = f"Refused: you can give between 1 and {self.budget_left(giver) or 0} message(s) of reading budget."
        elif to in self.no_read:
            err = f"Refused: Participant {to} cannot read messages written by others."
        elif self.cap and self.read_budget[to] + n > self.cap:
            op = "give_capped"
            err = (f"Refused: no participant may hold more than {self.cap} message(s) of reading budget in all, "
                   f"messages already read included; Participant {to} already has {self.read_budget[to]}.")
        if err:
            self.events.append({"op": op, "step": step, "from": giver, "to": to, "n": n, "error": err, **extra})
            return err
        self.read_budget[giver] -= n
        self.read_budget[to] += n
        self.events.append({"op": "give", "step": step, "from": giver, "to": to, "n": n, **extra})
        return None

    def allocate(self, moves: dict[int, int], step: int = -1) -> dict:
        snap = dict(self.read_budget)
        got: dict[int, int] = {}
        applied, refused = [], []
        for giver in sorted(moves):
            to = int(moves[giver])
            err, op = None, "give_refused"
            if giver not in snap or snap[giver] - len(self.read.get(giver, set())) < 1:
                err = ("Refused: your reading is not limited, so you have no budget to give." if giver not in snap
                       else "Refused: you have no reading budget to give.")
            elif to == giver:
                err = "Refused: give to another participant."
            elif to not in snap:
                err = f"Refused: Participant {to} can already read every message."
            elif to in self.no_read:
                err = f"Refused: Participant {to} cannot read messages written by others."
            elif self.cap and snap[to] + got.get(to, 0) + 1 > self.cap:
                op = "give_capped"
                err = (f"Refused: no participant may hold more than {self.cap} message(s) of reading budget in all, "
                       f"messages already read included; Participant {to} already has {snap[to] + got.get(to, 0)}.")
            if err:
                self.events.append({"op": op, "step": step, "from": giver, "to": to, "n": 1, "error": err, "round": "alloc"})
                refused.append({"from": giver, "to": to, "error": err})
                continue
            got[to] = got.get(to, 0) + 1
            applied.append([giver, to])
            self.events.append({"op": "give", "step": step, "from": giver, "to": to, "n": 1, "round": "alloc"})
        for giver, to in applied:
            self.read_budget[giver] -= 1
            self.read_budget[to] += 1
        return {"applied": applied, "refused": refused}

    def _admit(self, reader: int | None, ids: list[int]) -> tuple[list[int], int]:
        if reader is None or reader not in self.read_budget:
            return ids, 0
        seen = self.read.setdefault(reader, set())
        out, held = [], 0
        for i in ids:
            if self.messages[i].author == reader or i in seen or self._free(i):
                out.append(i)
            elif len(seen) < self.read_budget[reader]:
                seen.add(i)
                out.append(i)
            else:
                held += 1
        return out, held

    def _free(self, msg_id: int) -> bool:
        return self.free_read is not None and bool(self.free_read(self.messages[msg_id]))

    def _visible(self, msg_id: int, reader: int | None) -> bool:
        if not (0 <= msg_id < len(self.messages)):
            return False
        return (not self.private) or reader is None or self.messages[msg_id].author == reader

    @staticmethod
    def normalize_refs(refs) -> list[str]:
        """Accept a list or a comma/space separated string; canonicalise MessageN; de-duplicate in order."""
        if refs is None:
            return []
        if isinstance(refs, str):
            refs = [x for x in re.split(r"[,\s]+", refs) if x]
        seen, norm = set(), []
        for r in refs:
            r = str(r).strip().strip("'\"")
            m = MESSAGE_RE.match(r)
            if m:
                r = f"Message{int(m.group(1))}"
            if r and r not in seen:
                seen.add(r)
                norm.append(r)
        return norm

    def repair_refs(self, author: int, rationale: str, refs) -> list[str]:
        """Keep only references that satisfy both write rules (used for a forced final answer)."""
        keep = []
        for r in self.normalize_refs(refs)[: self.max_refs]:
            m = MESSAGE_RE.match(r)
            exists = self._visible(int(m.group(1)), author) if m else bool(self.resource_exists(r))
            if exists and _names_ref(rationale or "", r):
                keep.append(r)
        return keep

    def validate_refs(self, author: int, rationale: str, refs: Iterable[str]) -> tuple[list[str], str | None]:
        """Check the two write rules."""
        norm = self.normalize_refs(refs)
        if len(norm) > self.max_refs:
            return norm, f"A message carries at most {self.max_refs} references; got {len(norm)}."
        for r in norm:
            m = MESSAGE_RE.match(r)
            exists = self._visible(int(m.group(1)), author) if m else bool(self.resource_exists(r))
            if not exists:
                return norm, f"Rejected: reference '{r}' does not exist (only reference names you have actually seen)."
        for r in norm:
            if not _names_ref(rationale or "", r):
                return norm, (
                    f"Rejected: the rationale does not name reference '{r}' and what it contributed "
                    f"(e.g. \"{r} gave ...\")."
                )
        return norm, None

    def publish(self, author: int, sub_task: str, artifact: str, rationale: str,
                refs: Iterable[str], step: int) -> PublishResult:
        sub_task, artifact, rationale = (sub_task or "").strip(), (artifact or "").strip(), (rationale or "").strip()
        norm, err = self.validate_refs(author, rationale, refs)
        if err is None and self.extra_validator is not None:
            err = self.extra_validator(author, sub_task, artifact, rationale, norm)
        if err is not None:
            self.events.append({"op": "publish_rejected", "step": step, "author": author, "error": err})
            return PublishResult(False, error=err)
        note, truncated = None, False
        used = count_tokens(sub_task)
        if used + count_tokens(artifact) > self.token_limit:
            truncated = True
            if used >= self.token_limit:
                sub_task = truncate_to_tokens(sub_task, self.token_limit)
                artifact = ""
            else:
                artifact = truncate_to_tokens(artifact, self.token_limit - used)
            note = f"sub_task and artifact exceeded {self.token_limit} tokens and were truncated."
        msg = Message(
            id=len(self.messages), sub_task=sub_task, artifact=artifact, rationale=rationale,
            refs=norm, author=author, step=step, weight=self.s0, truncated=truncated,
        )
        for r in norm:
            m = MESSAGE_RE.match(r)
            if m:
                target = self.messages[int(m.group(1))]
                msg.msg_refs.append(target.id)
                target.cited_by.append(msg.id)
                target.weight += self.alpha * (1.0 - target.weight)
        self.messages.append(msg)
        self.index.add(f"{sub_task}\n{artifact}\n{rationale}")
        self.events.append({"op": "publish", "step": step, "author": author, "id": msg.id,
                            "msg_refs": list(msg.msg_refs), "n_refs": len(norm)})
        return PublishResult(True, message=msg, note=note)

    def decay(self) -> None:
        for m in self.messages:
            m.weight *= self.gamma

    def search(self, query: str, reader: int | None, step: int, k: int | None = None) -> list[Message]:
        k = k or self.k
        cand = [m.id for m in self.messages if self._visible(m.id, reader)]
        rel = self.index.scores(query or "", cand if self.private else None)
        ranked: list[tuple[float, int]] = []
        if rel:
            max_rel = max(rel.values())
            ids = sorted(rel)
            if self.ranking == "standard":
                ranked = [(self.messages[i].weight + rel[i] / max_rel, i) for i in ids]
            elif self.ranking == "bm25":
                ranked = [(rel[i] / max_rel, i) for i in ids]
            else:
                perm = ids[:]
                self.rng.shuffle(perm)
                ranked = [(float(len(perm) - j), i) for j, i in enumerate(perm)]
            ranked.sort(key=lambda t: (-t[0], t[1]))
        top = ranked[:k]
        headers: list[int] = []
        if self.choice and reader in self.read_budget:
            held = 0
            headers = [i for _, i in top if not self.known(reader, i)]
            shown = [(s, i) for s, i in top if i not in set(headers)]
        else:
            admitted, held = self._admit(reader, [i for _, i in top])
            shown = top = [(s, i) for s, i in top if i in set(admitted)]
        self.events.append({
            "op": "search", "step": step, "reader": reader, "query": query,
            "results": [{"id": i, "score": round(s, 6), "rel": round(rel[i] / max(rel.values()), 6),
                         "weight": round(self.messages[i].weight, 6)} for s, i in shown],
            "n_matched": len(ranked), **({"withheld": held} if held else {}), **({"headers": headers} if headers else {}),
        })
        return [self.messages[i] for _, i in top]

    def known(self, reader: int | None, msg_id: int) -> bool:
        """Whether `reader` may see this message's content now without spending budget."""
        return (reader not in self.read_budget or self.messages[msg_id].author == reader
                or msg_id in self.read.get(reader, set()) or self._free(msg_id) or msg_id in self.pushed.get(reader, set()))

    def log_push(self, reader: int, ids: list[int], step: int, view: str) -> None:
        if view == "full":
            self.pushed.setdefault(reader, set()).update(ids)
        self.events.append({"op": "push", "step": step, "reader": reader, "ids": list(ids), "view": view})

    def visit(self, names: Iterable[str], reader: int | None, step: int) -> tuple[list[Message], list[str]]:
        found, errors = [], []
        for raw in list(names)[: self.k]:
            raw = str(raw).strip()
            m = MESSAGE_RE.match(raw) or re.match(r"^(\d+)$", raw)
            if m and self._visible(int(m.group(1)), reader):
                if self._admit(reader, [int(m.group(1))])[0]:
                    found.append(self.messages[int(m.group(1))])
                else:
                    errors.append(f"{raw}: withheld (your reading budget is used up)")
            else:
                errors.append(f"{raw}: no such message")
        self.events.append({"op": "visit", "step": step, "reader": reader,
                            "ids": [m.id for m in found], "errors": errors})
        return found, errors

    def render(self, msg: Message, reader: int | None = None) -> str:
        if self.choice and reader in self.read_budget and not self.known(reader, msg.id):
            return (f"=== {msg.name} ===\n{self.describe(msg)}\n(not read: visit it to read its content; "
                    "that uses one message of your reading budget)")
        lines = [f"=== {msg.name} ===", f"sub_task: {msg.sub_task}", f"artifact: {msg.artifact}",
                 f"rationale: {msg.rationale}", f"refs: {', '.join(msg.refs) if msg.refs else '(none)'}"]
        known = (lambda i: True) if reader not in self.read_budget else (lambda i: self.known(reader, i))
        built_on = [self.messages[i] for i in msg.msg_refs if known(i)]
        if built_on:
            lines.append("builds on: " + "; ".join(f"{m.name} ({_short(m.sub_task)})" for m in built_on))
        citers = [self.messages[i] for i in msg.cited_by if self._visible(i, reader) and known(i)]
        citers = list(reversed(citers))[: self.max_citers_shown]
        if citers:
            lines.append("referenced by: " + "; ".join(f"{m.name} ({_short(m.sub_task)})" for m in citers))
        return "\n".join(lines)

    def render_many(self, msgs: list[Message], reader: int | None = None) -> str:
        if not msgs:
            return "(no messages)"
        return "\n\n".join(self.render(m, reader) for m in msgs)

    def evidence_chain(self, refs: Iterable[str]) -> list[int]:
        """Transitive closure of message references, starting from `refs`."""
        stack, seen = [], set()
        for r in refs:
            m = MESSAGE_RE.match(str(r).strip())
            if m and int(m.group(1)) < len(self.messages):
                stack.append(int(m.group(1)))
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            stack.extend(self.messages[i].msg_refs)
        return sorted(seen)

    def dump(self) -> dict:
        return {"messages": [asdict(m) for m in self.messages], "events": self.events,
                "config": {"alpha": self.alpha, "gamma": self.gamma, "k": self.k, "ranking": self.ranking,
                           "private": self.private, "token_limit": self.token_limit, **({"cap": self.cap} if self.cap else {})}}
