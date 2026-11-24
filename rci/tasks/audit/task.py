"""Contradiction Audit: find pairs of passages in a documentation corpus that contradict each other."""
from __future__ import annotations

import json
import re
import random
from pathlib import Path

from rci import paths
from rci.agents.tooling import INT, STR, STR_LIST, AgentCtx, ToolResult, ToolSpec
from rci.tasks.base import Task

SNIP_RE = re.compile(r"^Snippet(\d+)$")

SYSTEM = """You are auditing the documentation of a software project for factual contradictions.
The documentation has been split into numbered snippets (Snippet0, Snippet1, ...). Some pairs of
snippets make claims that cannot both be true: a parameter, default, unit, range, behaviour, version
or requirement is described one way in one place and differently in another. Most snippets are
consistent; your job is to find as many contradicting pairs as you can.

Tools:
- fetch_snippet(index): the full text of one snippet; its first line names the file it comes from.
- search_snippet(pattern): exact, case-insensitive text search over all snippets; returns up to ten
  matching snippets with the context around each match.

A contradiction is reported as a pair of snippet indices with a short reason. When you are done, call
submit_answer with the list of pairs you are confident of. Reports that are not real contradictions
are not useful; reports you never submit are lost."""


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class ContradictionAudit(Task):
    name = "audit"
    msg_token_limit = 256
    max_actions = 250
    max_steps = 250
    summarize = True

    def __init__(self, root: str = "", level: int = 4, projects: list[str] | None = None,
                 partition: str | None = None, strict: bool = False, share_hint: bool = False, f1: bool = False,
                 penalty: float | None = None):
        assert partition in (None, "snippet", "file"), partition
        self.root = Path(root) if root else paths.data_dir("audit", "eval")
        self.level, self.projects, self.partition = level, projects, partition
        self.strict = bool(strict or partition or f1 or penalty is not None)
        self.f1, self.penalty = f1, (None if penalty is None else float(penalty))
        self.share_hint = share_hint
        self._cache: dict[str, dict] = {}
        self._shares: dict[tuple[str, int], dict[int, int]] = {}

    def shares(self, example: dict, n: int) -> dict[int, int]:
        """snippet index -> share in 0..n-1 (all 0 without a partition or for n = 1)."""
        key = (example["id"], n)
        if key not in self._shares:
            sn = example["snippets"]
            if not self.partition or n <= 1:
                self._shares[key] = {s["idx"]: 0 for s in sn}
            else:
                rng = random.Random(f"{example['id']}|{n}|{self.partition}")
                if self.partition == "snippet":
                    order = [s["idx"] for s in sn]
                    rng.shuffle(order)
                    self._shares[key] = {idx: j % n for j, idx in enumerate(order)}
                else:
                    by_file: dict[str, list[int]] = {}
                    for s in sn:
                        by_file.setdefault(s["file"], []).append(s["idx"])
                    files = sorted(by_file)
                    rng.shuffle(files)
                    files.sort(key=lambda f: -len(by_file[f]))
                    load, out = [0] * n, {}
                    for f in files:
                        k = min(range(n), key=lambda i: (load[i], i))
                        load[k] += len(by_file[f])
                        out.update({idx: k for idx in by_file[f]})
                    self._shares[key] = out
        return self._shares[key]

    def _visible(self, ctx: AgentCtx, example: dict, idx: int) -> bool:
        if not self.partition or ctx.run is None:
            return True
        n = ctx.run.cfg.n_agents
        return n <= 1 or self.shares(example, n).get(idx) == ctx.agent_id % n

    def examples(self, limit: int | None = None) -> list[dict]:
        d = self.root / f"x{self.level}"
        files = sorted(d.glob("*.json"))
        out = []
        for f in files:
            inst = json.loads(f.read_text())
            if self.projects and inst["project"] not in self.projects:
                continue
            inst.setdefault("id", f"{inst['project']}_x{self.level}")
            self._cache[inst["id"]] = inst
            out.append(inst)
        return out[:limit] if limit else out

    def system_prompt(self, example: dict) -> str:
        return SYSTEM

    def task_prompt(self, example: dict, agent_id: int, n_agents: int, rng) -> str:
        n = len(example["snippets"])
        if self.partition and n_agents > 1:
            mine = sorted(i for i, k in self.shares(example, n_agents).items() if k == agent_id % n_agents)
            return (f"Project: {example['project']}. The corpus has {n} snippets, Snippet0 to Snippet{n - 1}, split "
                    f"among {n_agents} agents. You can open and search only your share: {len(mine)} snippets "
                    f"({', '.join(f'Snippet{i}' for i in mine)}); the other agents hold the rest. Many contradictions "
                    f"pair a snippet of yours with one held by another agent."
                    + (" To find those, tell the other agents what your snippets say, and use what they share."
                       if self.share_hint else "")
                    + f" Your starting position is Snippet{mine[rng.randrange(len(mine))]}. "
                    f"Find contradicting pairs of snippets.")
        start = rng.randrange(n)
        return (f"Project: {example['project']}. The corpus has {n} snippets, Snippet0 to Snippet{n - 1}. "
                f"Your starting position is Snippet{start}. Find contradicting pairs of snippets.")

    def _snip(self, example: dict, idx) -> dict | None:
        try:
            i = int(str(idx).replace("Snippet", ""))
        except ValueError:
            return None
        sn = example["snippets"]
        return sn[i] if 0 <= i < len(sn) else None

    def tools(self, example: dict, run) -> list[ToolSpec]:
        snippets = example["snippets"]
        lowered = [s["text"].lower() for s in snippets]

        def fetch(ctx: AgentCtx, args: dict) -> ToolResult:
            s = self._snip(example, args.get("index"))
            if s is None:
                return ToolResult(f"Error: no snippet {args.get('index')} (valid: 0..{len(snippets) - 1}).", ok=False)
            if not self._visible(ctx, example, s["idx"]):
                return ToolResult(f"Snippet{s['idx']} is held by another agent; you can open only your share.",
                                  ok=False)
            return ToolResult(f"Snippet{s['idx']}\n{s['text']}")

        def search(ctx: AgentCtx, args: dict) -> ToolResult:
            pat = str(args.get("pattern", "")).strip()
            if not pat:
                return ToolResult("Error: empty pattern.", ok=False)
            p = pat.lower()
            hits = []
            for s, low in zip(snippets, lowered):
                if not self._visible(ctx, example, s["idx"]):
                    continue
                j = low.find(p)
                if j >= 0:
                    a, b = max(0, j - 200), min(len(low), j + len(p) + 200)
                    hits.append(f"Snippet{s['idx']} ({s['file']}): ...{s['text'][a:b]}...")
                if len(hits) >= 10:
                    break
            if not hits:
                return ToolResult(f"No snippet contains '{pat}'.")
            return ToolResult(f"{len(hits)} matching snippets for '{pat}':\n\n" + "\n\n".join(hits))

        return [
            ToolSpec("fetch_snippet", "Return the full text of a snippet by index.",
                     {"type": "object", "properties": {"index": {**INT, "description": "Snippet index"}},
                      "required": ["index"]}, fetch),
            ToolSpec("search_snippet", "Exact, case-insensitive text search over the corpus (up to 10 hits).",
                     {"type": "object", "properties": {"pattern": {**STR, "description": "Text to find"}},
                      "required": ["pattern"]}, search),
        ]

    def blackboard_tools(self, example: dict, run, datastore) -> list[ToolSpec]:
        def report(ctx: AgentCtx, args: dict) -> ToolResult:
            a, b = self._snip(example, args.get("snippet_a")), self._snip(example, args.get("snippet_b"))
            if a is None or b is None or a["idx"] == b["idx"]:
                return ToolResult("Rejected: snippet_a and snippet_b must be two different valid snippets.", ok=False)
            qa, qb = str(args.get("quote_a", "")), str(args.get("quote_b", ""))
            for q, s in ((qa, a), (qb, b)):
                err = self.quote_error(q, s)
                if err:
                    return ToolResult(err, ok=False)
            na, nb = f"Snippet{a['idx']}", f"Snippet{b['idx']}"
            extra = [r for r in datastore.normalize_refs(args.get("refs")) if r not in (na, nb)]
            rationale = f'{na} says "{_norm_ws(qa)}"; {nb} says "{_norm_ws(qb)}". ' + str(args.get("rationale", ""))
            res = datastore.publish(ctx.agent_id, f"I've found a contradiction between {na} and {nb}",
                                    str(args.get("reason", "")), rationale, [na, nb] + extra, ctx.run.step)
            if not res.ok:
                return ToolResult(res.error, ok=False)
            return ToolResult(f"Filed as {res.message.name}. Remember to include ({a['idx']}, {b['idx']}) in your final answer.")

        return [ToolSpec(
            "report_contradiction",
            "Publish a contradiction you found as a message; both quotes must be verbatim text of their snippets.",
            {"type": "object", "properties": {
                "snippet_a": INT, "snippet_b": INT,
                "quote_a": {**STR, "description": "Verbatim conflicting text from snippet_a"},
                "quote_b": {**STR, "description": "Verbatim conflicting text from snippet_b"},
                "reason": {**STR, "description": "Why the two cannot both be true"},
                "refs": {**STR_LIST, "description": "Other messages this builds on (optional)"},
                "rationale": {**STR, "description": "Name each extra ref and what it contributed (optional)"},
            }, "required": ["snippet_a", "snippet_b", "quote_a", "quote_b", "reason"]}, report)]

    def quote_error(self, quote: str, s: dict) -> str | None:
        """None if `quote` is acceptable evidence from snippet `s`, else the rejection message."""
        if len(_norm_ws(quote)) < 8 or _norm_ws(quote) not in _norm_ws(s["text"]):
            return f"Rejected: the quote '{quote[:80]}' is not verbatim text of Snippet{s['idx']}."
        return None

    def resource_exists(self, example: dict, run, ref: str) -> bool:
        m = SNIP_RE.match(ref)
        return bool(m) and int(m.group(1)) < len(example["snippets"])

    def answer_properties(self):
        return {"pairs": {"type": "array", "description": "Contradicting pairs you are confident of.",
                          "items": {"type": "object", "properties": {
                              "a": INT, "b": INT, "reason": STR}, "required": ["a", "b"]}}}, ["pairs"]

    def recorded_answer(self, answers: list[dict]) -> dict | None:
        pairs = [p for a in (answers or []) for p in ((a or {}).get("pairs") or [])]
        return {"pairs": pairs} if pairs else None

    @staticmethod
    def _pairs(final: dict | None) -> list[tuple[int, int]]:
        out = []
        for p in (final or {}).get("pairs") or []:
            try:
                out.append((int(p.get("a")), int(p.get("b"))))
            except (TypeError, ValueError, AttributeError):
                continue
        return out

    def score_agent(self, example: dict, final: dict | None) -> float:
        return self._agent_score(example, self._pairs(final))

    def _agent_score(self, example: dict, pairs: list[tuple[int, int]]) -> float:
        if self.penalty is not None:
            return self._bet(example, pairs)
        return self._f1(example, pairs) if self.f1 else self._fraction(example, pairs)

    def _bet(self, example: dict, pairs: list[tuple[int, int]]) -> float:
        reported = {frozenset(p) for p in pairs if p[0] != p[1]}
        true = {frozenset((c["edited"], c["partner"])) for c in example["contradictions"]}
        hit = len(reported & true)
        return (hit - self.penalty * (len(reported) - hit)) / max(1, len(true))

    def _f1(self, example: dict, pairs: list[tuple[int, int]]) -> float:
        reported = {frozenset(p) for p in pairs if p[0] != p[1]}
        true = {frozenset((c["edited"], c["partner"])) for c in example["contradictions"]}
        hit = len(reported & true)
        if not hit:
            return 0.0
        prec, rec = hit / len(reported), hit / max(1, len(true))
        return 2 * prec * rec / (prec + rec)

    def _fraction(self, example: dict, pairs: list[tuple[int, int]]) -> float:
        if self.strict:
            reported_pairs = {frozenset(p) for p in pairs}
            found = sum(1 for c in example["contradictions"]
                        if frozenset((c["edited"], c["partner"])) in reported_pairs)
        else:
            reported = {x for p in pairs for x in p}
            found = sum(1 for c in example["contradictions"] if c["edited"] in reported)
        return found / max(1, len(example["contradictions"]))

    def score_group(self, example: dict, finals: list[dict], run=None) -> dict:
        """Union over agents: a contradiction is found when some reported pair includes its edited snippet."""
        per, union = [], []
        for f in finals:
            pairs = self._pairs(f.get("final"))
            per.append(None if f.get("failing") else self._agent_score(example, pairs))
            if not f.get("failing"):
                union.extend(pairs)
        n_reported = len(set(tuple(sorted(p)) for p in union))
        true = {frozenset((c["edited"], c["partner"])) for c in example["contradictions"]}
        reported = {frozenset(p) for p in union if p[0] != p[1]}
        n_exact = len(reported & true)
        out = {"score": self._fraction(example, union), "per_agent": per, "n_reported_pairs": n_reported,
               "n_exact_pairs": n_exact, "precision": n_exact / len(reported) if reported else None}
        if self.penalty is not None:
            out["recall"], out["f1"] = out["score"], self._f1(example, union)
            out["score"], out["penalty"] = self._bet(example, union), self.penalty
        elif self.f1:
            out["recall"] = out["score"]
            out["score"] = self._f1(example, union)
        if self.strict:
            out["strict"] = True
        return out
