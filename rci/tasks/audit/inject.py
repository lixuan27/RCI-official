"""LLM stage of Contradiction Audit construction: inject contradictions, judge them, build levels."""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from rci import paths
from rci.agents.llm import LLMClient
from rci.agents.tokens import count_tokens, truncate_to_tokens

SUBJECT_PROMPT = """Below are two snippets from the documentation of {project}, each shown inside the file it
comes from (the snippet itself is between <<<SNIPPET>>> markers).

Name up to three subjects that BOTH snippets discuss and that are central to what their files document
(a concrete option, function, parameter, default, behaviour, limit, rule or procedure), not passing
mentions or generic topics. If there is no such shared subject, return an empty list.

Return JSON only: {{"subjects": ["...", "..."]}}

=== File A: {file_a} ===
{ctx_a}

=== File B: {file_b} ===
{ctx_b}"""

EDIT_PROMPT = """You are preparing a documentation-audit benchmark. Two snippets from the {project} documentation
discuss a shared subject: {subjects}.

Rewrite the prose of ONE of the two snippets so that it can no longer be true together with the other
snippet: change a fact about the shared subject (a value, default, unit, range, condition, behaviour,
ordering, requirement or capability) so the two snippets now conflict.
Rules:
- Edit ONLY natural-language prose sentences. Never change code, code blocks, commands, tables, list
  items, headings, or links.
- Change only a few sentences; copy everything else in the snippet EXACTLY, character for character.
- The edited snippet must still read naturally on its own and stay consistent with itself.
- The conflict must be detectable only by reading both snippets, not from general knowledge of
  {project} alone; prefer specific details stated in the other snippet.

Answer in exactly this format (no JSON):
EDIT: A or B
WHY: <one sentence: why the two now cannot both be true>
NEW_TEXT:
<<<
<the complete rewritten snippet, including its first line>
>>>

=== Snippet A ({file_a}) ===
{text_a}

=== Snippet B ({file_b}) ===
{text_b}"""

JUDGE_PROMPT = """You are checking one item of a documentation-audit benchmark. Snippet E was edited so that it
contradicts Snippet P. Answer four questions strictly.

1. natural: Does Snippet E read naturally on its own, like real documentation (no artifacts, no
   obviously inserted or awkward text)?
2. nontrivial: Would a reader need Snippet P to notice the problem, i.e. is the edited claim NOT
   something a model could flag as false from memorized general knowledge of {project} alone?
3. self_consistent: Is Snippet E consistent with itself (no internal contradiction)?
4. conflict: Do Snippet E and Snippet P truly make claims that cannot both be true?

Claimed conflict: {why}

=== Snippet E ({file_e}) ===
{text_e}

=== Snippet P ({file_p}) ===
{text_p}

Return JSON only: {{"natural": true/false, "nontrivial": true/false, "self_consistent": true/false, "conflict": true/false, "note": "<short>"}}"""


def load_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_json(text: str) -> dict | None:
    """Lenient JSON extraction: tolerates code fences and raw newlines inside strings."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    for cand in (m.group(0), m.group(0).replace("\\'", "'")):
        try:
            return json.loads(cand, strict=False)
        except json.JSONDecodeError:
            continue
    return None


_FENCE = re.compile(r"^\s*(```|~~~)")
_NONPROSE = re.compile(r"^\s*([-*+]\s|\d+[.)]\s|#\.?\s|#{1,6}\s|\||\+[-=+]{3,}|[=\-~^`'\"*+#]{3,}\s*$|\.\. |:\w+:)")


def nonprose_lines(text: str) -> set[int]:
    """Indices of lines that are code, tables, list items, headings or directives."""
    lines = text.split("\n")
    bad, in_fence, literal_indent = set(), False, None
    for i, ln in enumerate(lines):
        if _FENCE.match(ln):
            in_fence = not in_fence
            bad.add(i)
            continue
        if in_fence:
            bad.add(i)
            continue
        if literal_indent is not None:
            if ln.strip() == "" or (len(ln) - len(ln.lstrip())) > literal_indent:
                bad.add(i)
                continue
            literal_indent = None
        if _NONPROSE.match(ln) or ln.startswith("    ") or ln.startswith("\t") or i == 0:
            bad.add(i)
        if i + 1 < len(lines) and re.match(r"^\s*[=\-~^`'\"*+#]{3,}\s*$", lines[i + 1]) and ln.strip():
            bad.add(i)
        if ln.rstrip().endswith("::"):
            literal_indent = len(ln) - len(ln.lstrip())
    return bad


def fenced_lines(text: str) -> set[int]:
    """Lines inside ``` / ~~~ fenced code blocks (including the fences)."""
    out, inside = set(), False
    for i, ln in enumerate(text.split("\n")):
        if _FENCE.match(ln):
            inside = not inside
            out.add(i)
        elif inside:
            out.add(i)
    return out


def mechanical_check(orig: str, new: str, corpus_texts: list[str], self_idx: int) -> tuple[bool, str, dict]:
    a, b = orig.split(), new.split()
    if not b:
        return False, "empty", {}
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    changed = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
    frac = changed / max(1, len(a))
    info = {"changed_frac": round(frac, 3), "n_places": len(ops)}
    if not ops:
        return False, "no change", info
    if frac > 0.30:
        return False, "changed > 30%", info
    if len(ops) > 8:
        return False, "more than 8 change places", info
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    if norm(orig) == norm(new):
        return False, "formatting-only change", info
    ol, nl = orig.split("\n"), new.split("\n")
    if not nl or nl[0] != ol[0]:
        return False, "file header line changed", info
    lm = difflib.SequenceMatcher(None, ol, nl, autojunk=False)
    fence_o, fence_n = fenced_lines(orig), fenced_lines(new)
    changed_new_lines = []
    for tag, i1, i2, j1, j2 in lm.get_opcodes():
        if tag == "equal":
            continue
        if any(i in fence_o for i in range(i1, i2)) or any(j in fence_n for j in range(j1, j2)):
            return False, "edit touches a fenced code block", info
        changed_new_lines.extend(nl[j1:j2])
    others = "\n".join(t for k, t in enumerate(corpus_texts) if k != self_idx)
    for ln in changed_new_lines:
        for sent in re.split(r"(?<=[.!?])\s+", ln.strip()):
            s = sent.strip()
            if len(s) < 12:
                continue
            if s in others or new.count(s) != 1:
                return False, "edited sentence not unique in corpus", info
    return True, "ok", info


def parse_edit(text: str) -> dict | None:
    """Parse the tagged edit format; falls back to JSON for robustness."""
    text = text or ""
    m_e = re.search(r"EDIT:\s*\**\s*([AB])\b", text)
    m_t = re.search(r"NEW_TEXT:\s*<<<\n?(.*?)\n?>>>", text, flags=re.S)
    if m_e and m_t:
        m_w = re.search(r"WHY:\s*(.+)", text)
        return {"edit": m_e.group(1), "new_text": m_t.group(1), "why": (m_w.group(1).strip() if m_w else "")}
    return parse_json(text)


def file_context(snips_by_file: dict[str, list[dict]], snip: dict, limit: int = 6000) -> str:
    parts = []
    for s in snips_by_file[snip["file"]]:
        if s["idx"] == snip["idx"]:
            parts.append("<<<SNIPPET>>>\n" + s["text"] + "\n<<<SNIPPET>>>")
        else:
            parts.append(s["text"])
    ctx = "\n\n".join(parts)
    if count_tokens(ctx) <= limit:
        return ctx
    i = ctx.index("<<<SNIPPET>>>")
    left, right = ctx[:i], ctx[i:]
    right = truncate_to_tokens(right, limit // 2 + count_tokens(snip["text"]))
    keep_left = max(0, limit - count_tokens(right))
    left_ids = left[-keep_left * 4:] if keep_left else ""
    return "...\n" + left_ids + right + "\n..."


class Injector:
    def __init__(self, llm: LLMClient, judge_llm: LLMClient | None, project: str, split: str,
                 per_kind: int = 10, batch: int = 24, max_candidates: int = 6000, seed: int = 0,
                 second_pass: bool = False):
        self.llm, self.judge_llm = llm, judge_llm or llm
        self.project, self.split, self.per_kind, self.batch = project, split, per_kind, batch
        d = paths.data_dir("audit") / split / project
        self.dir = d
        self.snips = load_jsonl(d / "snippets.jsonl")
        self.by_id = {s["id"]: s for s in self.snips}
        self.texts = [s["text"] for s in self.snips]
        self.by_file: dict[str, list[dict]] = defaultdict(list)
        for s in self.snips:
            self.by_file[s["file"]].append(s)
        for v in self.by_file.values():
            v.sort(key=lambda s: s["idx"])
        self.cands = load_jsonl(d / "candidates.jsonl")[:max_candidates]
        self.out_path = d / "contradictions.json"
        self.log_path = d / "inject_log.jsonl"
        self.rng = random.Random(seed)
        self.accepted: list[dict] = []
        self.used: set[int] = set()
        if self.out_path.exists():
            prev = json.loads(self.out_path.read_text())
            self.accepted = prev["contradictions"]
            self.used = {x for c in self.accepted for x in (c["edited"], c["partner"])}
        self.done_ranks = set()
        if self.log_path.exists():
            last = {}
            for l in open(self.log_path):
                if l.strip():
                    d = json.loads(l)
                    last[d["rank"]] = d
            retry = lambda d: d["stage"] == "edit" and "finish" not in d or (
                d["stage"] == "mechanical" and d.get("reason") == "edit touches code/table/list/heading")
            self.done_ranks = {r for r, d in last.items() if not retry(d)}
            if second_pass:
                tried_twice = {r for r, n in __import__("collections").Counter(
                    json.loads(l)["rank"] for l in open(self.log_path) if l.strip()).items() if n > 1}
                self.done_ranks = {r for r in self.done_ranks
                                   if last[r]["stage"] not in ("edit", "mechanical", "judge") or r in tried_twice}

    def need(self, same_file: bool) -> bool:
        return sum(1 for c in self.accepted if c["same_file"] == same_file) < self.per_kind

    async def ask(self, llm: LLMClient, prompt: str, tag: str, max_tokens: int = 12000) -> dict | None:
        call = await llm.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=1.0,
                              tag=tag, enable_thinking=True)
        return parse_json(call.content)

    async def try_pair(self, cand: dict) -> dict:
        a, b = self.by_id[cand["a"]], self.by_id[cand["b"]]
        rec = {"rank": cand["rank"], "a": a["idx"], "b": b["idx"], "same_file": cand["same_file"]}
        sub = await self.ask(self.llm, SUBJECT_PROMPT.format(
            project=self.project, file_a=a["file"], file_b=b["file"],
            ctx_a=file_context(self.by_file, a), ctx_b=file_context(self.by_file, b)), "subject", 6000)
        subjects = [s for s in (sub or {}).get("subjects", []) if isinstance(s, str) and s.strip()][:3]
        if not subjects:
            return {**rec, "stage": "subject", "ok": False}
        call = await self.llm.chat([{"role": "user", "content": EDIT_PROMPT.format(
            project=self.project, subjects="; ".join(subjects), file_a=a["file"], file_b=b["file"],
            text_a=a["text"], text_b=b["text"])}], max_tokens=30000, temperature=1.0, tag="edit", enable_thinking=True)
        ed = parse_edit(call.content)
        if not ed or ed.get("edit") not in ("A", "B") or not ed.get("new_text"):
            return {**rec, "stage": "edit", "ok": False, "subjects": subjects, "finish": call.finish_reason,
                    "content_chars": len(call.content or ""), "error": call.error}
        e, p = (a, b) if ed["edit"] == "A" else (b, a)
        new = ed["new_text"]
        hdr = e["text"].split("\n", 1)[0]
        if not new.startswith(hdr):
            new = hdr + "\n" + new.lstrip("\n")
        ok, why_bad, info = mechanical_check(e["text"], new, self.texts, e["idx"])
        rec.update({"subjects": subjects, "edited": e["idx"], "partner": p["idx"], "why": ed.get("why", ""), **info})
        if not ok:
            return {**rec, "stage": "mechanical", "ok": False, "reason": why_bad}
        votes = await asyncio.gather(*(self.ask(self.judge_llm, JUDGE_PROMPT.format(
            project=self.project, why=ed.get("why", ""), file_e=e["file"], text_e=new, file_p=p["file"],
            text_p=p["text"]), f"judge{j}", 8000) for j in range(4)))
        keys = ("natural", "nontrivial", "self_consistent", "conflict")
        passed = [bool(v) and all(v.get(k) is True for k in keys) for v in votes]
        rec["judges"] = [{k: (v or {}).get(k) for k in keys} for v in votes]
        if not all(passed):
            return {**rec, "stage": "judge", "ok": False}
        return {**rec, "stage": "accepted", "ok": True, "original": e["text"], "new_text": new,
                "file_edited": e["file"], "file_partner": p["file"]}

    async def run(self) -> dict:
        t0 = time.time()
        pending = [c for c in self.cands if c["rank"] not in self.done_ranks]
        i = 0
        while (self.need(True) or self.need(False)) and i < len(pending):
            batch = []
            while len(batch) < self.batch and i < len(pending):
                c = pending[i]
                i += 1
                ia, ib = self.by_id[c["a"]]["idx"], self.by_id[c["b"]]["idx"]
                if ia in self.used or ib in self.used or not self.need(c["same_file"]):
                    continue
                if any({ia, ib} & {self.by_id[x["a"]]["idx"], self.by_id[x["b"]]["idx"]} for x in batch):
                    continue
                batch.append(c)
            if not batch:
                continue
            results = await asyncio.gather(*(self.try_pair(c) for c in batch))
            with open(self.log_path, "a") as f:
                for r in sorted(results, key=lambda r: r["rank"]):
                    slim = {k: v for k, v in r.items() if k not in ("original", "new_text")}
                    f.write(json.dumps(slim) + "\n")
                    if r["ok"] and self.need(r["same_file"]) and r["edited"] not in self.used \
                            and r["partner"] not in self.used:
                        self.accepted.append(r)
                        self.used |= {r["edited"], r["partner"]}
            self.save()
            n_same = sum(c["same_file"] for c in self.accepted)
            print(f"[inject] {self.project}: tried {i}/{len(pending)} accepted {len(self.accepted)} "
                  f"(same {n_same}, cross {len(self.accepted) - n_same}) {time.time() - t0:.0f}s", flush=True)
        return {"project": self.project, "n": len(self.accepted)}

    def save(self) -> None:
        tmp = self.out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"project": self.project, "split": self.split,
                                   "contradictions": self.accepted}, indent=1))
        tmp.replace(self.out_path)


def build_levels(split: str, levels: list[int], seed: int = 0, projects: list[str] | None = None) -> None:
    base = paths.data_dir("audit") / split
    for d in sorted(p for p in base.iterdir() if (p / "contradictions.json").exists()):
        proj = d.name
        if projects and proj not in projects:
            continue
        snips = load_jsonl(d / "snippets.jsonl")
        con = json.loads((d / "contradictions.json").read_text())["contradictions"]
        if len(con) < 20:
            print(f"[levels] {proj}: only {len(con)} contradictions, skipping")
            continue
        con = con[:20]
        keep = {x for c in con for x in (c["edited"], c["partner"])}
        edited_text = {c["edited"]: c["new_text"] for c in con}
        rng = random.Random(seed)
        pool_whole = [s["idx"] for s in snips if s["idx"] not in keep and s.get("whole_file")]
        pool_rest = [s["idx"] for s in snips if s["idx"] not in keep and not s.get("whole_file")]
        rng.shuffle(pool_whole)
        rng.shuffle(pool_rest)
        order = pool_whole + pool_rest
        for n in levels:
            extra = order[: 40 * (n - 1)]
            chosen = sorted(keep | set(extra))
            remap = {old: new for new, old in enumerate(chosen)}
            by_idx = {s["idx"]: s for s in snips}
            inst = {"id": f"{proj}_x{n}", "project": proj, "level": n, "snippets": [
                {"idx": remap[i], "file": by_idx[i]["file"], "text": edited_text.get(i, by_idx[i]["text"])}
                for i in chosen],
                "contradictions": [{"edited": remap[c["edited"]], "partner": remap[c["partner"]],
                                    "same_file": c["same_file"], "why": c.get("why", "")} for c in con]}
            out = base / f"x{n}"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{proj}.json").write_text(json.dumps(inst))
            print(f"[levels] {proj} x{n}: {len(chosen)} snippets")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.audit.inject",
                                 description="Inject and judge contradictions, then build the audit levels.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inject")
    a.add_argument("--split", default="eval")
    a.add_argument("--projects", default=None)
    a.add_argument("--base-url", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--judge-base-url", default=None)
    a.add_argument("--judge-model", default=None)
    a.add_argument("--per-kind", type=int, default=10)
    a.add_argument("--batch", type=int, default=24)
    a.add_argument("--parallel-projects", type=int, default=4)
    a.add_argument("--second-pass", action="store_true", help="retry pairs that failed after the subject step")
    l = sub.add_parser("levels")
    l.add_argument("--split", default="eval")
    l.add_argument("--levels", default="1,2,4,8,16,32")
    l.add_argument("--projects", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "levels":
        build_levels(args.split, [int(x) for x in args.levels.split(",")],
                     projects=args.projects.split(",") if args.projects else None)
        return
    llm = LLMClient(args.base_url.split(","), args.model, max_concurrency=512)
    jllm = LLMClient(args.judge_base_url.split(","), args.judge_model, max_concurrency=512) \
        if args.judge_base_url else None
    projects = args.projects.split(",") if args.projects else sorted(
        p.name for p in (paths.data_dir("audit") / args.split).iterdir() if (p / "candidates.jsonl").exists())

    async def go():
        sem = asyncio.Semaphore(args.parallel_projects)

        async def one(p):
            async with sem:
                return await Injector(llm, jllm, p, args.split, per_kind=args.per_kind, batch=args.batch,
                                      second_pass=args.second_pass).run()
        print(await asyncio.gather(*(one(p) for p in projects)))
    asyncio.run(go())


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
