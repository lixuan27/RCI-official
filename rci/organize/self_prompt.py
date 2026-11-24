"""Self-written prompts: notes a model writes for its own group, from the rules or evolved over generations."""
from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

NOTES_WORDS = 150
FWD = "I've forwarded facts I hold, word for word"

RULES = """Each task is a group decision among {n} participants. Each participant holds some facts the others do not; the correct choice shows only when every participant's facts are taken into account, and a participant who decides from part of them is easily misled. Participants share facts through a common store that only takes facts copied word for word until a participant has read every participant's facts; such a participant may then post an analysis, which the store delivers to everyone. Each participant can read only {budget} message(s) written by others, and may give unused reading budget to another participant. Until someone who has read every participant's facts posts an analysis, the others must wait before answering (the hold lifts after a while). The group's decision is its members' answers. Nobody is told whether an answer was right."""

SHARED_SYSTEM = """You write the notes that every participant of the next group reads before a task. Participants are numbered 0 to {last} and keep their numbers from task to task.

{rules}

Below is how the last group worked on each of its tasks. You are not told which answers were right; judge only from how the group worked. Write notes that would help the next group decide from all participants' facts: say concretely what each participant should do and when, drawn from what happened. Do not describe personalities.

Reply with the notes between <notes> and </notes>, at most {words} words."""

RULES_ONLY_SYSTEM = """You write the notes that every participant of a group reads before a task. Participants are numbered 0 to {last}.

{rules}

Write notes that would help the group decide from all participants' facts: say concretely what each participant should do and when. Do not describe personalities.

Reply with the notes between <notes> and </notes>, at most {words} words."""

GIVE_SENTENCE = ("Each participant can read only {budget} message(s) written by others, and may give unused reading "
                 "budget to another participant.")
ALLOC_SENTENCE = ("Each participant can read only {budget} message(s) written by others. Before reading opens, each "
                  "participant decides once, at the same time as the others, whether to keep its reading budget or give "
                  "it to one other participant; the result is shown to everyone, and there is no other way to give.")
CAP_SENTENCE = (" No participant can hold more than {cap} message(s) of reading budget in all, its own included; a gift "
                "that would go over this is refused.")
DELIVERS = "such a participant may then post an analysis, which the store delivers to everyone."
RELAY_SENTENCE = (" Facts may be forwarded word for word both from what a participant was given and from other "
                  "participants' messages it has read, alone or together with its own.")

DS_K, DS_ACTIONS, DS_ROUNDS, DS_NOTE_CAP = 24, 12, 24, 6000
DS_COMPARISON = ("The group's answer will be compared with independent members who spend the same tokens and with a "
                 "single agent that spends the same tokens.")
RULES_DS = """Each task is a question about a corpus of {k} documents, numbered 1 to {k}, of about 1,400 tokens each. {n} participants, numbered 0 to {last}, work on the same task with the same corpus, and every participant can open every document. {member} Each participant has {actions} actions and {rounds} rounds; each tool call uses one action unless it is free, and every tool result ends with the actions and rounds left. Each participant's own tools: `open_doc` returns a document's full text (one action; at most 2 per turn); `search` returns at most 8 lines that contain a pattern, each with its document number and cut to at most 160 characters and 48 tokens (one action; at most 4 per turn); `note` keeps a private note (free; at most {note_cap:,} tokens of notes in all, the oldest dropped beyond that); the submission tool submits the participant's own answer with its confidence (one action). A call over a per-turn limit is refused and uses no action.{claim} `post_digest` posts, for the others, digests of documents the poster has opened itself with `open_doc` (one action per call): each digest holds only quotes copied word for word from one document, 1 to 8 quotes of 20 to 300 characters each, at most 1,800 characters and 400 tokens in all and at most 450 tokens with its header, and no other text; the host checks every quote and refuses a digest that fails; the board keeps only the first verified digest of each document, and a later digest of the same document is refused as a duplicate, at no cost. `post_analysis` posts an analysis or recommendation in free text of at most 1,500 tokens (one action; a longer one is refused, at no cost); only a participant that has covered every document (opened it itself, or received its verified digest) can post one, and only the group's first analysis is accepted, later ones being refused at no cost. Delivery: {what} delivered once to every other participant, at its next call, as a message kept in its history; the analysis comes under the heading "posted by a participant who has covered every document".{plan}{wait} {compaction} Each participant submits its own answer; the group commits to the answer the most members give, ties broken by their stated confidence, and a participant's answer is submitted for it when it reaches its action or round limit. Nobody is told whether an answer was right. {comparison}"""
DS_RULES_CLAIM = (" `claim` tells the others which documents a participant will read (free; at most 3 claims per "
                  "participant, each of 1 to 24 distinct document numbers).")
DS_RULES_PLAN = " Reading opens at round {k}: before that round, `open_doc` and `search` are refused, at no cost."
DS_RULES_WAIT = (" Timing rule: a participant that has not covered every document cannot submit until the first "
                 "analysis has reached it, or until round {k}; a held submission costs no action.")
DS_SHARED_SYSTEM = """You write the notes that every participant of the next group reads before a task. Participants are numbered 0 to {last} and keep their numbers from task to task.

{rules}

Below is how the last group worked on each of its tasks and what it spent. You are not told which answers were right; judge only from how the group worked. Rewrite the notes to keep the verified organisation while spending fewer tokens: say concretely what each participant should do and when, drawn from what happened. Do not describe personalities.

Reply with the notes between <notes> and </notes>, at most {words} words."""
DS_RULES_SYSTEM = RULES_ONLY_SYSTEM.replace("decide from all participants' facts", "answer correctly")
DS_PLAN, DS_WAIT, DS_PER_GEN, DS_GENS = 2, 16, 16, 4


def rules_text(n: int, budget: int, cap: int = 0, relay: bool = False, alloc: bool = False) -> str:
    """The HiddenBench blackboard rules as the notes writer reads them."""
    text = RULES.format(n=n, budget=budget)
    give = GIVE_SENTENCE.format(budget=budget)
    if alloc:
        text, give = text.replace(give, ALLOC_SENTENCE.format(budget=budget)), ALLOC_SENTENCE.format(budget=budget)
    if cap:
        text = text.replace(give, give + CAP_SENTENCE.format(cap=cap))
    if relay:
        text = text.replace(DELIVERS, DELIVERS + RELAY_SENTENCE)
    return text


def relays_by(rec: dict, row: dict) -> Counter:
    """Per author, its forwards that hold another seat's private fact."""
    from rci.tasks.hiddenbench.facts import holders

    n = len(rec["agents"])
    return Counter(m["author"] for m in rec["method_state"].get("messages", [])
                   if m["sub_task"] == FWD and any(h != m["author"] for h in holders(row, m["artifact"], n)[0]))


def episode_summary(rec: dict, members: list, tag: str, row: dict | None = None, cap: bool = False) -> str:
    """How the group worked on one task, from its record alone; no correct answer appears."""
    ms = rec["method_state"]
    ev = ms.get("events", [])
    msgs = {m["id"]: m for m in ms.get("messages", [])}
    n = len(rec["agents"])
    lines = [f"Task {tag}:"]
    gifts = [e for e in ev if e["op"] == "give"]
    refused = [e for e in ev if e["op"] == "give_refused"]
    if gifts:
        lines.append("  gifts: " + "; ".join(f"round {e['step']}: Participant {e['from']} gave {e['n']} to "
                                             f"Participant {e['to']}" for e in gifts))
    if refused:
        lines.append("  refused gifts: " + "; ".join(f"round {e['step']}: Participant {e['from']} to {e['to']} "
                                                     f"({e['error'][9:60]})" for e in refused[:6]))
    capped = [e for e in ev if e["op"] == "give_capped"] if cap else []
    if capped:
        lines.append("  gifts refused by the cap: " + "; ".join(f"round {e['step']}: Participant {e['from']} to {e['to']}"
                                                              for e in capped[:6]))
    if not gifts and not refused and not capped:
        lines.append("  nobody gave reading budget.")
    analyses = sorted((m for m in msgs.values() if m["sub_task"] != FWD), key=lambda m: (m["step"], m["id"]))
    if analyses:
        a0 = analyses[0]
        lines.append(f"  first analysis: Participant {a0['author']} at round {a0['step']} (it had read every "
                     f"participant's facts)")
    else:
        lines.append("  no analysis was posted: nobody read every participant's facts.")
    holds = Counter(h["agent"] for h in ms.get("holds") or [] if h.get("kind") in ("submit", "wait", "frozen"))
    preds = [str(x) if x else "(none)" for x in (rec.get("score") or {}).get("preds") or []]
    relays = relays_by(rec, row) if row is not None else None
    for m in members:
        j = m["agent"]
        fw = sum(1 for x in msgs.values() if x["author"] == j and x["sub_task"] == FWD)
        state = "held every fact" if m["complete"] else "did not hold every fact"
        extra = "" if relays is None else f" ({relays.get(j, 0)} passing on other participants' facts)"
        if cap:
            extra += f", had {sum(e['from'] == j for e in capped)} gift(s) refused by the cap"
        lines.append(f"  Participant {j}: forwarded {fw} message(s){extra}, read {len(m['shown'])} by others, {state}, "
                     f"waited {holds.get(j, 0)} time(s), answered at round {m['last']}: "
                     f"{preds[j] if j < len(preds) else '(none)'}")
    c = Counter(p for p in preds if p != "(none)")
    if c:
        _, k = c.most_common(1)[0]
        lines.append(f"  answers: {k} of {n} gave the same answer" + (" (all agreed)" if k == n else ""))
    return "\n".join(lines)


def shared_prompt(n: int, budget: int, notes: str, record: str, **rules) -> list:
    user = f"Current notes:\n{notes or '(none yet)'}\n\n{record}"
    return [{"role": "system", "content": SHARED_SYSTEM.format(last=n - 1, rules=rules_text(n, budget, **rules),
                                                               words=NOTES_WORDS)},
            {"role": "user", "content": user}]


def rules_prompt(n: int, budget: int, **rules) -> list:
    return [{"role": "system", "content": RULES_ONLY_SYSTEM.format(last=n - 1, rules=rules_text(n, budget, **rules),
                                                                   words=NOTES_WORDS)},
            {"role": "user", "content": "Write the notes."}]


def parse(text: str, tag: str, old: str) -> str:
    """The text between the last pair of tags, cut to NOTES_WORDS words; `old` when there is none."""
    m = re.findall(rf"<{tag}>(.*?)</{tag}>", text or "", re.S)
    if not m or not m[-1].strip():
        return old
    return " ".join(m[-1].split()[:NOTES_WORDS])


def ds_rules_text(n: int = 4, claims: bool = True, *, planning_steps: int = DS_PLAN, wait_until: int = DS_WAIT) -> str:
    """The DocSweep group rules as the notes writer reads them."""
    from rci.tasks.docsweep import task as ds_task

    blocks = tuple(b for b in ds_task.GROUP_BLOCKS if claims or b != "claims")
    what = ("each new claim, each newly verified digest and the first analysis is" if claims else
            "each newly verified digest and the first analysis is")
    return RULES_DS.format(
        k=DS_K, n=n, last=n - 1, member=ds_task.MEMBER_SENTENCE, actions=DS_ACTIONS, rounds=DS_ROUNDS,
        note_cap=DS_NOTE_CAP, claim=DS_RULES_CLAIM if claims else "", what=what,
        plan=DS_RULES_PLAN.format(k=planning_steps + 1) if planning_steps else "",
        wait=DS_RULES_WAIT.format(k=wait_until + 1) if wait_until else "",
        compaction=ds_task.compaction_rule(blocks), comparison=DS_COMPARISON)


def ds_rules_prompt(n: int = 4, **rules) -> list:
    return [{"role": "system", "content": DS_RULES_SYSTEM.format(last=n - 1, rules=ds_rules_text(n, **rules),
                                                                 words=NOTES_WORDS)},
            {"role": "user", "content": "Write the notes."}]


def ds_shared_prompt(n: int, notes: str, record: str, **rules) -> list:
    user = f"Current notes:\n{notes or '(none yet)'}\n\n{record}"
    return [{"role": "system", "content": DS_SHARED_SYSTEM.format(last=n - 1, rules=ds_rules_text(n, **rules),
                                                                  words=NOTES_WORDS)},
            {"role": "user", "content": user}]


def _ranges(docs) -> str:
    xs, out, i = sorted(set(int(d) for d in docs)), [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ", ".join(out)


def ds_episode_summary(rec: dict, tag: str = "", k: int = DS_K) -> str:
    """How a DocSweep group worked and what it spent on one task; no answer or score appears."""
    ms = rec.get("method_state") or {}
    agents = rec.get("agents") or []
    lines = [f"Task {tag}:" if tag else "Task:"]
    claims = ms.get("claims") or []
    if claims:
        lines.append("  claims: " + "; ".join(f"round {c['step'] + 1}: Participant {c['author']} claimed documents "
                                              f"{_ranges(c['docs'])}" for c in claims))
    else:
        lines.append("  nobody claimed documents.")
    opened = [[e["doc"] for e in ((a.get("task_state") or {}).get("ds") or {}).get("open_log") or []] for a in agents]
    by_doc = Counter(d for docs in opened for d in set(docs))
    shared = sum(1 for c in by_doc.values() if c > 1)
    repeats = sum(len(docs) - len(set(docs)) for docs in opened)
    lines.append(f"  documents opened by the group: {len(by_doc)} of {k}; opened by more than one participant: "
                 f"{shared}; repeated opens of a document by the same participant: {repeats}.")
    refusals = [r for r in ms.get("refusals") or [] if r.get("tool") == "post_digest"]
    rejected = Counter(r["verdict"] for r in refusals if r.get("verdict") != "duplicate")
    dups = sum(1 for r in refusals if r.get("verdict") == "duplicate")
    rej = sum(rejected.values())
    lines.append(f"  digests: {len(ms.get('board') or {})} verified; {rej} rejected"
                 + (" (" + ", ".join(f"{v}: {c}" for v, c in sorted(rejected.items())) + ")" if rej else "")
                 + f"; {dups} duplicate(s) refused.")
    a0 = ms.get("analysis")
    lines.append(f"  first analysis: Participant {a0['author']} at round {a0['step'] + 1}." if a0 else
                 "  no analysis was posted.")
    cov = ms.get("coverage") or {}
    forced = 0
    for i, a in enumerate(agents):
        c = cov.get(i, cov.get(str(i))) or {}
        docs = opened[i]
        rep = len(docs) - len(set(docs))
        parts = [f"opened {len(set(docs))} document(s)" + (f" ({rep} repeated open(s))" if rep else "")]
        parts.append(f"covered every document from round {c['covered_step'] + 1}" if c.get("covered_step") is not None
                     else "never covered every document")
        if c.get("covered_at_submit") is not None:
            parts.append(f"had covered {c['covered_at_submit']} of {k} when it submitted")
        if c.get("submit_step") is not None:
            parts.append(f"submitted at round {c['submit_step'] + 1}" + (" (forced by the host)"
                                                                        if a.get("final_forced") else ""))
        forced += bool(a.get("final_forced"))
        parts.append(f"compacted its history {a.get('compactions', 0)} time(s)")
        parts.append(f"{sum(int(x.get('completion_tokens') or 0) for x in a.get('calls') or []):,} completion tokens")
        lines.append(f"  Participant {i}: " + ", ".join(parts) + ".")
    total = int((rec.get("compute") or {}).get("completion_tokens") or 0)
    lines.append(f"  forced submissions: {forced} of {len(agents)}.")
    lines.append(f"  completion tokens: {total:,} in the episode, {total // max(1, len(agents)):,} per participant on "
                 f"average.")
    return "\n".join(lines)


def call_seed(key: str) -> int:
    """A sampling seed fixed by the call's key, so that two notes calls differ and are reproducible."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


async def write_call(llm, messages: list, key: str, max_tokens: int, seeded: bool = False) -> tuple:
    c = await llm.chat(messages, tools=None, max_tokens=max_tokens, temperature=1.0, enable_thinking=True,
                       tag="reflect", route_key=key, seed=call_seed(key) if seeded else None)
    if c.error:
        raise SystemExit(f"the notes call failed: {c.error}")
    return c.content or "", {"completion_tokens": c.completion_tokens, "prompt_tokens": c.prompt_tokens}


def write_new(path: Path, doc: dict) -> str:
    """Write a notes file once; frozen notes are never overwritten."""
    if path.exists():
        raise SystemExit(f"{path} exists; frozen notes are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1) + "\n").encode("utf-8")
    with open(path, "xb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def batches(examples: list, k: int, generations: int, seed: int) -> list:
    order = list(examples)
    random.Random(seed).shuffle(order)
    return [[order[(g * k + i) % len(order)] for i in range(k)] for g in range(generations)]


def stats(recs: list, rows: dict, cap: bool = False) -> dict:
    """Label-free counts of a generation: analyses, gifts, members holding every fact, full agreement."""
    from rci.tasks.hiddenbench.facts import members

    out = Counter()
    for r in recs:
        ms = r["method_state"]
        ev = ms.get("events", [])
        out["records"] += 1
        out["with_analysis"] += any(m["sub_task"] != FWD for m in ms.get("messages", []))
        out["with_gift"] += any(e["op"] == "give" for e in ev)
        out["refused_gifts"] += sum(e["op"] == "give_refused" for e in ev)
        out["members_complete"] += sum(m["complete"] for m in members(r, rows[str(r["example_id"])]))
        preds = [p for p in (r.get("score") or {}).get("preds") or [] if p]
        out["all_agree"] += bool(preds) and len(set(preds)) == 1 and len(preds) == len(r["agents"])
        if cap:
            out["give_capped"] += sum(e["op"] == "give_capped" for e in ev)
    return dict(out)


async def hiddenbench(a) -> None:
    from rci.agents.llm import LLMClient
    from rci.agents.runner import GroupRun, RunConfig
    from rci.registry import make_protocol, make_task
    from rci.run import out_path, write_atomic
    from rci.tasks.hiddenbench.facts import members

    out = Path(a.out)
    work = Path(a.work) if a.work else out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    rules = {"cap": a.cap, "relay": a.relay}
    llm = LLMClient(a.base_urls.split(","), a.model, max_concurrency=max(a.concurrency * a.n * 2, 64))
    if a.kind == "rules":
        text, usage = await write_call(llm, rules_prompt(a.n, a.budget, **rules), f"rules/{a.seed}", a.max_tokens)
        notes = parse(text, "notes", "")
        if not notes:
            raise SystemExit("the reply holds no <notes>; nothing written")
        write_new(out, {"kind": "rules", "notes": notes})
        return
    task_kw, kw = json.loads(a.task_kwargs), json.loads(a.protocol_kwargs)
    if a.cap and kw.setdefault("budget_cap", a.cap) != a.cap:
        raise SystemExit(f"--cap {a.cap} but --protocol-kwargs sets budget_cap {kw['budget_cap']}")
    if a.relay and not task_kw.setdefault("relay_doc", True):
        raise SystemExit("--relay but --task-kwargs sets relay_doc false")
    task = make_task("hiddenbench", **task_kw)
    examples = task.examples()
    rows = {str(e["id"]): e for e in examples}
    plan = batches(examples, a.per_generation, a.generations, a.seed)
    for g in range(a.generations):
        path = work / f"notes_g{g}.json"
        if not path.exists():
            if g:
                raise SystemExit(f"{path} is missing")
            path.write_text(json.dumps({"kind": "shared", "notes": ""}))
        cur = json.loads(path.read_text())
        root = work / f"g{g}"
        sem = asyncio.Semaphore(a.concurrency)
        cell = {**kw, "prompt_file": str(path)}

        async def one(ex, root=root, cell=cell):
            protocol = make_protocol("blackboard", **cell)
            p = out_path(root, protocol.name, a.n, a.seed, ex["id"])
            if p.exists():
                with gzip.open(p, "rt") as fh:
                    return json.load(fh)
            cfg = RunConfig(n_agents=a.n, max_actions=a.max_actions, max_steps=a.max_steps, gen_limit=16384,
                            summarize=False, seed=a.seed, method_kwargs=cell)
            async with sem:
                try:
                    rec = await GroupRun(task, ex, protocol, cfg, llm).run()
                except Exception as e:
                    print(f"[self_prompt] g{g} {ex['id']} failed: {e!r}", flush=True)
                    return None
            write_atomic(p, rec)
            return rec

        recs = [r for r in await asyncio.gather(*(one(e) for e in plan[g])) if r is not None]
        nxt = work / f"notes_g{g + 1}.json"
        if nxt.exists():
            continue
        record = "\n\n".join(episode_summary(r, members(r, rows[str(r["example_id"])]), str(i + 1),
                                             row=rows[str(r["example_id"])] if a.relay else None, cap=bool(a.cap))
                             for i, r in enumerate(sorted(recs, key=lambda r: str(r["example_id"]))))
        text, usage = await write_call(llm, shared_prompt(a.n, a.budget, cur["notes"], record, **rules),
                                       f"reflect/g{g}", a.max_tokens)
        tmp = nxt.with_suffix(".tmp")
        tmp.write_text(json.dumps({"kind": "shared", "notes": parse(text, "notes", cur["notes"])}, indent=1))
        tmp.replace(nxt)
        with open(work / "log.jsonl", "a") as f:
            f.write(json.dumps({"generation": g, "stats": stats(recs, rows, bool(a.cap)), **usage}) + "\n")
    final = json.loads((work / f"notes_g{a.generations}.json").read_text())
    write_new(out, {"kind": "shared", "notes": final["notes"]})


def ds_generation(records: str, protocol_dir: str, n: int = 4) -> list:
    out = []
    for f in sorted(glob.glob(os.path.join(records, protocol_dir, f"N{n}", "s*", "*.json*"))):
        if f.endswith((".json", ".json.gz")):
            with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
                out.append(json.load(fh))
    return sorted(out, key=lambda r: (str(r.get("example_id")), int((r.get("config") or {}).get("seed", 0))))


async def docsweep(a) -> None:
    from rci.agents.llm import LLMClient

    llm = LLMClient(a.base_urls.split(","), a.model, max_concurrency=4)
    if a.command == "rules":
        msgs = ds_rules_prompt(4)
        key = f"docsweep/rules_{a.which}"
    else:
        cur = json.loads(Path(a.notes).read_text())
        recs = ds_generation(a.records, a.protocol_dir)
        train = json.loads(Path(a.questions).read_text()).get("train") or []
        want = train[DS_PER_GEN * a.gen: DS_PER_GEN * (a.gen + 1)]
        if len(recs) != DS_PER_GEN or sorted(str(r.get("example_id")) for r in recs) != sorted(want):
            raise SystemExit(f"generation {a.gen} is incomplete or holds other tasks ({len(recs)} records)")
        record = "\n\n".join(ds_episode_summary(r, tag=str(i + 1)) for i, r in enumerate(recs))
        msgs = ds_shared_prompt(4, cur.get("notes") or "", record)
        key = f"docsweep/{a.chain}/g{a.gen}"
    text, _ = await write_call(llm, msgs, key, a.max_tokens, seeded=True)
    notes = parse(text, "notes", "")
    if not notes:
        raise SystemExit("the reply holds no <notes>; nothing written")
    write_new(Path(a.out), {"kind": "shared" if a.command == "evolve" else "rules", "notes": notes})


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write a group's prompt with the members' own model: from the rules alone, "
                                             "or evolved over generations from unlabeled records.")
    sub = ap.add_subparsers(dest="bench", required=True)
    h = sub.add_parser("hiddenbench")
    h.add_argument("--kind", required=True, choices=["rules", "evolved"])
    h.add_argument("--task-kwargs", default="{}")
    h.add_argument("--protocol-kwargs", default="{}", help="blackboard kwargs of the protocol the notes are for")
    h.add_argument("--budget", type=int, default=1, help="reading budget per member, as the rules state it")
    h.add_argument("--cap", type=int, default=0, help="reading-budget cap stated in the rules and set on the protocol")
    h.add_argument("--relay", action="store_true", help="state that facts may be relayed")
    h.add_argument("--n", type=int, default=4)
    h.add_argument("--generations", type=int, default=4)
    h.add_argument("--per-generation", type=int, default=16)
    h.add_argument("--max-actions", type=int, default=16)
    h.add_argument("--max-steps", type=int, default=24)
    h.add_argument("--concurrency", type=int, default=16)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--work", default=None, help="directory of the generations (default: --out without its suffix)")
    d = sub.add_parser("docsweep")
    dsub = d.add_subparsers(dest="command", required=True)
    r = dsub.add_parser("rules")
    r.add_argument("--which", required=True, help="a label that fixes the sampling seed, e.g. A or B")
    e = dsub.add_parser("evolve")
    e.add_argument("--chain", required=True)
    e.add_argument("--gen", required=True, type=int)
    e.add_argument("--records", required=True, help="results root of the generation's runs")
    e.add_argument("--protocol-dir", default="docsweep_board")
    e.add_argument("--notes", required=True, help="the notes the generation ran with")
    e.add_argument("--questions", required=True, help="DocSweep questions file with its 'train' list")
    for p in (h, r, e):
        p.add_argument("--base-urls", default=os.environ.get("RCI_BASE_URLS", ""))
        p.add_argument("--model", required=True)
        p.add_argument("--max-tokens", type=int, default=32768)
        p.add_argument("--out", required=True, help="the notes file to write (never overwritten)")
    a = ap.parse_args(argv)
    if not a.base_urls:
        ap.error("--base-urls (or RCI_BASE_URLS) is required")
    asyncio.run(hiddenbench(a) if a.bench == "hiddenbench" else docsweep(a))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
