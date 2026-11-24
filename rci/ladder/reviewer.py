"""Reviewer that selects one of the members' answers, and a replay command for finished cells."""
from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rci.agents.agent import Agent, AgentConfig
from rci.agents.tooling import STR, AgentCtx, ToolResult, ToolSpec
from rci.ladder.commit import _candidates, _vote_key, vote_stake
from rci.tasks.base import SUBMIT_NAME, stake

LABELS = "12345678"
MODES = ("read", "verify", "patch", "integrate")

REVIEW_SYSTEM = """You are the reviewer of a group of agents. Each member worked on the task below and submitted an answer, and they do not all agree. You do not answer from scratch: you decide which of the candidate answers is correct, and the group commits to your choice.

How many members back a candidate and how confident they say they are are weak signals: members often repeat the same mistake, and a popular answer can be wrong. Judge each candidate by its evidence.
{how}
Reject a candidate whose evidence does not hold or that leaves a condition of the task unmet, and prefer the candidate whose claims you can confirm. If no candidate can be confirmed, choose the most plausible one.

When you have decided, call `{submit}` with the candidate's number."""

HOW = {
    "read": "You have no access to the sources: weigh the reasoning each member gives, and look for claims that contradict the task or each other.",
    "verify": ("Check the claims against the sources with your tools: open what a candidate cites, confirm every condition of the task, and "
               "use your budget on the claims that separate the candidates. You have at most {actions} tool calls."),
    "patch": ("You have the repository at its original commit. Each candidate is a patch at /opt/review/candidate_<number>.diff: apply one with "
              "`git apply`, try it, and undo it with `git checkout -- . && git clean -fd` before the next. Write a small test that shows the issue, "
              "and run it and the existing tests near the change against the candidates that differ. Use your budget on what separates them. You "
              "have at most {actions} tool calls."),
}
INTEGRATE_SYSTEM = """You are the integrator of a group of agents. Each member worked on the task below on its own and left a patch. You ship the group's one patch: what is in /testbed when you call `{submit}` is what the group submits.

You have the repository at its original commit, and each candidate is a patch at /opt/review/candidate_<number>.diff. Apply one with `git apply`, undo it with `git checkout -- . && git clean -fd`, and try them against a small test that shows the issue and the existing tests near the change. Then leave /testbed in the best state you can reach: a candidate as it is, parts of several combined, or a candidate with what is wrong in it fixed. How many members back a patch is a weak signal: members often repeat the same mistake. You have at most {actions} tool calls, and your tests stay out of what you ship: remove them before you call `{submit}`.

"""
SHIP_NAME = "ship_patch"


@dataclass
class Candidate:
    label: str
    answer: str
    members: list[int]
    stake: float
    score: float
    notes: list[str] = field(default_factory=list)
    key: object = None


def submitted(rec: dict) -> list[int]:
    """Members who answered, without those forced at the action cap when anyone submitted on their own."""
    cand = _candidates(rec)
    free = [i for i in cand if not rec["finals"][i].get("forced")]
    return free or cand


def _note(final: dict, chars: int) -> str:
    parts = []
    for k, title in (("explanation", "Explanation"), ("rationale", "Reasoning")):
        v = str(final.get(k) or "").strip()
        if v:
            parts.append(f"{title}: {v[:chars]}")
    refs = [str(r) for r in (final.get("refs") or [])][:12]
    if refs:
        parts.append("Cites: " + ", ".join(refs))
    return "\n".join(parts) or "(no explanation given)"


def build_candidates(rec: dict, max_cands: int = 8, notes_per: int = 2, note_chars: int = 1500, seed: int = 0,
                     pool: list[int] | None = None) -> list[Candidate]:
    per = rec["score"]["per_agent"]
    finals = rec["finals"]
    pool = submitted(rec) if pool is None else pool
    clusters: dict = defaultdict(list)
    for i in pool:
        k = _vote_key(rec, i)
        clusters[k if k is not None else ("member", i)].append(i)

    def conf(i):
        return stake(finals[i]["final"])

    top = sorted(clusters.values(), key=lambda v: (len(v), sum(conf(i) for i in v)), reverse=True)[:min(max_cands, len(LABELS))]
    random.Random(f"{seed}/{rec.get('example_id')}").shuffle(top)
    out = []
    for lab, v in zip(LABELS, top):
        lead = sorted(v, key=conf, reverse=True)
        out.append(Candidate(lab, str(finals[lead[0]]["final"].get("answer", "")).strip(), v, sum(conf(i) for i in v),
                             sum(float(per[i]) for i in v) / len(v),
                             [_note(finals[i]["final"], note_chars) for i in lead[:notes_per]], _vote_key(rec, v[0])))
    return out


def patch_candidates(rec: dict, max_cands: int = 6, diff_chars: int = 6000, note_chars: int = 800, seed: int = 0) -> list[Candidate]:
    per, patches, finals = rec["score"]["per_agent"], rec["score"]["patches"], rec["finals"]
    groups: dict = defaultdict(list)
    for i, f in enumerate(finals):
        p = patches.get(str(i), "")
        if p.strip() and not f.get("failing"):
            groups[p].append(i)
    top = sorted(groups.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)[:min(max_cands, len(LABELS))]
    random.Random(f"{seed}/{rec.get('example_id')}").shuffle(top)
    out = []
    for lab, (patch, members) in zip(LABELS, top):
        notes = [f"```diff\n{patch[:diff_chars]}\n```"]
        summary = str((finals[members[0]].get("final") or {}).get("summary") or "").strip()
        if summary:
            notes.append(f"Its author's summary: {summary[:note_chars]}")
        out.append(Candidate(lab, patch, members, sum(stake(finals[i].get("final")) for i in members),
                             sum(float(per[i] or 0.0) for i in members) / len(members), notes))
    return out


def render_patches(cands: list[Candidate], joined: int) -> str:
    out = [f"## Candidate patches\n\n{joined} members worked on this and these are their distinct patches."]
    for c in cands:
        out.append(f"\n### Candidate {c.label}: from {len(c.members)} member{'s' if len(c.members) != 1 else ''}")
        out += ["\n" + n for n in c.notes]
    return "\n".join(out)


def render(cands: list[Candidate], joined: int) -> str:
    """The candidates as the reviewer reads them: answers, supporters, confidence, notes; no verdicts."""
    out = [f"## Candidate answers\n\n{joined} members answered."]
    for c in cands:
        out.append(f"\n### Candidate {c.label}: backed by {len(c.members)} member{'s' if len(c.members) != 1 else ''}, "
                   f"summed confidence {c.stake:.1f}\nAnswer: {c.answer}")
        out += [f"\nA member's account:\n{n}" for n in c.notes]
    return "\n".join(out)


class _Review:
    def __init__(self, example: dict):
        self.example, self.task, self.method, self.step, self.run_id = example, None, None, 0, f"review/{example['id']}"
        self.state: dict = {}


def choose_tool(cands: list[Candidate]) -> ToolSpec:
    labels = [c.label for c in cands]

    def fn(ctx: AgentCtx, args: dict) -> ToolResult:
        lab = str(args.get("candidate", "")).strip().upper().removeprefix("CANDIDATE").strip()
        if lab not in labels:
            return ToolResult(f"Error: candidate must be one of {labels}.", ok=False)
        return ToolResult("Choice recorded.", final={"candidate": lab})

    return ToolSpec(SUBMIT_NAME, "Commit the group to one candidate answer. This ends your review.",
                    {"type": "object", "properties": {"candidate": {**STR, "enum": labels, "description": "The candidate's number."}},
                     "required": ["candidate"]}, fn, terminal=True)


async def review(task, example: dict, rec: dict, llm, mode: str = "verify", max_actions: int = 24, seed: int = 0,
                 gen_limit: int = 8192, temperature: float = 1.0) -> dict:
    """One reviewer on one finished run."""
    if mode not in MODES:
        raise ValueError(mode)
    cands = patch_candidates(rec, seed=seed) if mode == "patch" else build_candidates(rec, seed=seed)
    joined = len(_candidates(rec))
    result = {"example_id": rec["example_id"], "mode": mode, "n_candidates": len(cands), "joined": joined,
              "candidates": [{"label": c.label, "answer": c.answer[:200], "n": len(c.members), "stake": c.stake, "score": c.score}
                             for c in cands],
              "chosen": cands[0].label if len(cands) == 1 else None, "reviewed": False, "actions": 0, "prompt_tokens": 0,
              "completion_tokens": 0}
    if len(cands) < 2:
        return result
    run = _Review(example)
    tools = ([] if mode == "read" else task.tools(example, run)) + [choose_tool(cands)]
    how = HOW[mode].format(actions=max_actions)
    system = REVIEW_SYSTEM.format(how=how, submit=SUBMIT_NAME)
    if mode != "read":
        system += "\n\n" + task.system_prompt(example)
    if mode == "patch":
        await asyncio.to_thread(task.stage_candidates, example, run, {c.label: c.answer for c in cands})
    shown = render_patches(cands, joined) if mode == "patch" else render(cands, joined)
    prompt = task.task_prompt(example, 0, 1, random.Random(seed)) + "\n\n" + shown
    agent = Agent(0, system, prompt, tools, llm, AgentConfig(max_actions=max_actions, gen_limit=gen_limit, temperature=temperature,
                                                             summarize=False), run)
    for step in range(max_actions + 4):
        run.step = step
        await agent.turn(step)
        if agent.done:
            break
    if not agent.done:
        await agent.force_final(max_actions + 4)
    if mode == "patch":
        await asyncio.to_thread(task.release, run)
    result.update(chosen=(agent.final or {}).get("candidate"), reviewed=True, actions=agent.actions,
                  prompt_tokens=sum(c.prompt_tokens for c in agent.calls),
                  completion_tokens=sum(c.completion_tokens for c in agent.calls))
    return result


def ship_tool() -> ToolSpec:
    def fn(ctx: AgentCtx, args: dict) -> ToolResult:
        return ToolResult("Shipped.", final={"note": str(args.get("note", ""))})

    return ToolSpec(SHIP_NAME, "Ship what is in /testbed as the group's patch. This ends your work.",
                    {"type": "object", "properties": {"note": {**STR, "description": "What you shipped, in a sentence."}}, "required": ["note"]},
                    fn, terminal=True)


async def integrate(task, example: dict, rec: dict, llm, max_actions: int = 60, seed: int = 0, gen_limit: int = 8192,
                    temperature: float = 1.0) -> dict:
    """An integrator on one finished run."""
    cands = patch_candidates(rec, seed=seed)
    result = {"example_id": rec["example_id"], "mode": "integrate", "n_candidates": len(cands), "joined": len(_candidates(rec)),
              "candidates": [{"label": c.label, "answer": c.answer[:200], "n": len(c.members), "stake": c.stake, "score": c.score}
                             for c in cands],
              "chosen": None, "reviewed": False, "actions": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if not cands:
        return result
    run = _Review(example)
    tools = task.tools(example, run) + [ship_tool()]
    system = INTEGRATE_SYSTEM.format(submit=SHIP_NAME, actions=max_actions) + task.system_prompt(example)
    await asyncio.to_thread(task.stage_candidates, example, run, {c.label: c.answer for c in cands})
    prompt = task.task_prompt(example, 0, 1, random.Random(seed)) + "\n\n" + render_patches(cands, result["joined"])
    agent = Agent(0, system, prompt, tools, llm, AgentConfig(max_actions=max_actions, gen_limit=gen_limit, temperature=temperature,
                                                             summarize=False), run)
    for step in range(max_actions + 4):
        run.step = step
        await agent.turn(step)
        if agent.done:
            break
    if not agent.done:
        await agent.force_final(max_actions + 4)
    patch = await task.snapshot(example, _Workspace(run))
    await asyncio.to_thread(task.release, run)
    result.update(reviewed=True, actions=agent.actions, prompt_tokens=sum(c.prompt_tokens for c in agent.calls),
                  completion_tokens=sum(c.completion_tokens for c in agent.calls), patch=patch[:200_000])
    if patch.strip():
        same = next((c for c in cands if c.answer.strip() == patch.strip()), None)
        score = same.score if same else float(bool((await asyncio.to_thread(task._evaluate, example, patch)).get("resolved")))
        result.update(shipped=float(score), chosen=same.label if same else "new", created=bool(score > 0.5 and all(c.score <= 0.5 for c in cands)))
    return result


class _Workspace:
    def __init__(self, run):
        self.state = run.state


def pooled(recs: list[dict]) -> dict:
    """The k runs of one question as one record, so a reviewer reads each distinct answer once."""
    finals = [f for r in recs for f in r["finals"]]
    per = [x for r in recs for x in r["score"]["per_agent"]]
    return {**recs[0], "finals": finals, "score": {**recs[0]["score"], "per_agent": per}}


async def review_label(task, example: dict, recs: list[dict], llm, max_actions: int = 16, seed: int = 0, gen_limit: int = 8192) -> tuple:
    rec = pooled(recs)
    cands = build_candidates(rec, seed=seed)
    if not cands:
        return None, False
    if len(cands) == 1:
        return cands[0].key, False
    res = await review(task, example, rec, llm, "verify", max_actions, seed, gen_limit)
    by = {c.label: c.key for c in cands}
    top = max(cands, key=lambda c: (len(c.members), c.stake))
    return by.get(res.get("chosen"), top.key), True


def pick(result: dict, fallback: float) -> tuple[float, bool]:
    """(the chosen candidate's score, whether a choice was made); without one, the commitment the group already had."""
    if "shipped" in result:
        return float(result["shipped"]), True
    for c in result["candidates"]:
        if c["label"] == result.get("chosen"):
            return float(c["score"]), True
    return float(fallback), False


def load_cell(cell: str) -> list:
    """(seed directory, example id, record) of every graded record of one cell directory."""
    out = []
    for f in sorted(glob.glob(os.path.join(cell, "s*", "*.json.gz"))):
        try:
            with gzip.open(f, "rt") as fh:
                r = json.load(fh)
        except (OSError, EOFError, ValueError):
            continue
        sc = r.get("score") or {}
        if sc.get("score") is None or not sc.get("per_agent") or not _candidates(r):
            continue
        out.append((Path(f).parent.name, str(r["example_id"]), r))
    return out


def free_commit(rec: dict) -> float:
    """vote_stake over the members who submitted on their own."""
    pool = set(submitted(rec))
    finals = rec["finals"]
    kept = {**rec, "finals": [f if i in pool else {**f, "final": None} for i, f in enumerate(finals)]}
    return float(vote_stake(kept))


def summary(runs: list, out: Path) -> dict:
    """Paired comparison of the reviewer's choice with the group's own commitment and with pass@N."""
    rows = []
    for seed, ex, rec in runs:
        p = out / f"{seed}_{ex}.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        base = float(vote_stake(rec))
        got, made = pick(res, base)
        per = [float(rec["score"]["per_agent"][i]) for i in _candidates(rec)]
        rows.append((base, free_commit(rec), got, max(per), res["n_candidates"], made, bool(res["reviewed"]),
                     res["actions"], res["prompt_tokens"] + res["completion_tokens"]))
    n = len(rows)
    if n < 2:
        print(f"{n} runs replayed")
        return {"runs": n}

    def ms(i, j=None):
        d = [r[i] - (r[j] if j is not None else 0) for r in rows]
        m = sum(d) / n
        return m, math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)

    rev = [r for r in rows if r[6]]
    out_ = {"runs": n, "reviewed": len(rev), "commit": ms(0)[0], "free": ms(1)[0], "reviewer": ms(2)[0],
            "pass": ms(3)[0], "reviewer_minus_commit": ms(2, 0), "reviewer_minus_free": ms(2, 1)}
    print(f"runs {n}, reviewed {len(rev)}, mean candidates {sum(r[4] for r in rows) / n:.1f}")
    print(f"commit {out_['commit']:.3f}  free {out_['free']:.3f}  reviewer {out_['reviewer']:.3f}  pass {out_['pass']:.3f}")
    print("reviewer - commit {:+.3f} +- {:.3f}".format(*out_["reviewer_minus_commit"]))
    print("reviewer - free   {:+.3f} +- {:.3f}".format(*out_["reviewer_minus_free"]))
    return out_


async def _replay(a) -> None:
    from rci.agents.llm import LLMClient
    from rci.registry import make_task

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cell = os.path.join(a.root, a.protocol, f"N{a.n}")
    runs = load_cell(cell)
    if a.limit:
        runs = runs[:a.limit]
    if a.command == "replay":
        task = make_task(a.task, **json.loads(a.task_kwargs))
        examples = {str(e["id"]): e for e in task.examples(limit=100000)}
        llm = LLMClient(a.base_urls.split(","), a.model, max_concurrency=max(a.concurrency * 2, 64))
        sem = asyncio.Semaphore(a.concurrency)

        async def one(seed: str, ex: str, rec: dict) -> None:
            p = out / f"{seed}_{ex}.json"
            if p.exists():
                return
            async with sem:
                try:
                    if a.mode == "integrate":
                        res = await integrate(task, examples[ex], rec, llm, a.max_actions, temperature=a.temperature)
                    else:
                        res = await review(task, examples[ex], rec, llm, a.mode, a.max_actions, temperature=a.temperature)
                except Exception as e:
                    print(f"[reviewer] {seed}/{ex} failed: {e!r}", flush=True)
                    return
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(res))
            tmp.replace(p)

        await asyncio.gather(*(one(s, e, r) for s, e, r in runs))
    res = summary(runs, out)
    (out / "summary.json").write_text(json.dumps(res, indent=1))


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay a reviewer on the finished records of one cell and compare it "
                                             "with the group's own commitment.")
    ap.add_argument("command", choices=["replay", "summary"])
    ap.add_argument("--root", required=True, help="results root of the cell (the rci.run --out directory)")
    ap.add_argument("--protocol", required=True, help="protocol directory name, including any tag")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--task", default="hiddenbench")
    ap.add_argument("--task-kwargs", default="{}")
    ap.add_argument("--mode", choices=MODES, default="verify")
    ap.add_argument("--base-urls", default=os.environ.get("RCI_BASE_URLS", ""))
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-actions", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.command == "replay" and (not a.base_urls or not a.model):
        ap.error("replay needs --base-urls and --model")
    asyncio.run(_replay(a))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
