"""A supervisor with the members' weights rewrites the debate team from unlabeled records, version by version."""
from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import median

from rci.protocols.team_notes import (FORM_TAG, HELD_PREFIX, KIND, LEAK_MIN_WORDS, LIMITS, N_SEATS, ROUNDS_ALLOWED,
                                      SUBMIT_TOOL, SUBMITTED_PREFIX, SUMMARY_EPISODES, check_team)
from rci.paths import load_benchmark
from rci.recursion.turns import first_turn_stats
from rci.tasks.hiddenbench.task import norm

EPISODE_CHARS = 600
DEV_TASKS = 48
INPUT_FIELDS = ("id", "description", "possible_answers", "shared_information", "hidden_information")
SCORE_KEYS = frozenset({"score", "per_agent", "preds", "correct", "correct_answer", "explanation", "full_answer",
                        "full_agree"})
SECTION = re.compile(r"^## v(\d+)\n", re.M)

RULES_TEXT = """Each task is a group decision among {n} participants, agent 0 to agent {last}; they keep their numbers from task to task. Each participant holds some facts the others do not. The group debates for R rounds, where R is set for the team before the task ({rounds}). In each round every participant takes one turn, at the same time as the others: it writes one text, which the others read under its number before their next turn, together with the option it named if it tried to submit. A participant's thinking before it writes is shown to nobody.{cut} Each participant submits its own answer with a tool, and only in the last round: a submission tried earlier is not accepted and reaches the others as that participant's tentative answer.{stated} A participant that has not submitted when the last round ends is made to submit at once. The host does not tell participants beforehand how many rounds there are; one whose early submission is refused is then told the number of rounds and which round it is in, and one that ends a turn without submitting is reminded to submit when it is confident.{asked}{hint}{decision} Nobody is told whether an answer was right.{comparison}"""
CONTENT_CAP_TOKENS = 1024
CUT_SENTENCE = (f" Of the text a participant writes in a turn, the others read at most the first {CONTENT_CAP_TOKENS:,} "
                "tokens.")
STATED_SENTENCE = (" A submission states the participant's confidence in its answer, and one that is not accepted reaches "
                   "the others with that confidence.")
ASKED_SENTENCE = (" From the second round on, the host puts before each participant's turn, after the others' texts, a "
                  "request to examine its own work and the others' step by step and to submit when it is confident; a "
                  "participant whose submission is not accepted is asked in reply to write its reasoning and its "
                  "current answer for the others.")
HINT_SENTENCE = (" Each participant's task ends by asking it, before it decides, to tell the others every fact it holds "
                 "that they may not have and to read what they have told it.")
DECISION_SENTENCE = " The group's decision is its members' answers."
COMPARISON_SENTENCE = " The group's result will be compared with independent members who spend the same tokens."

SUPERVISOR_ROLE = ("You are the supervisor of a team of {n} members. The members are copies of the same model as you; "
                   "you do not take part in their tasks.")
SUPERVISOR_JOB = (" After the team has worked through a batch of tasks you study how it worked and set up its next "
                  "version: how many rounds it debates and the notes its members read before every task.")
RULES_ONLY_JOB = (" You set the team up before it starts: how many rounds it debates and the notes its members read "
                  "before every task.")
SETTINGS = """What you set:
- rounds: {rounds}.
- shared notes, read by every member: at most {shared_words} words; they may be empty.
- one card for each member, agent 0 to agent {last}, read by that member only: at most {card_words} words each. The cards may be the same or give the members different parts.
A member reads the notes after its task's instructions and is told its own number with them."""
SUPERVISOR_GIVEN = ("Below you are given your learnings document so far, the record of the earlier versions, the current "
                    "version, statistics of the current version over all its tasks, and a closer account of some of "
                    "those tasks. Nobody knows which answers were right and you are not told: judge only from how the "
                    "team worked.")
RULES_ONLY_GIVEN = "You have no record of any team's work: reason from the rules alone."
WRITING = ("What you write goes to teams that work on other tasks: put no fact, name or option of any task into the "
           "notes or the learnings.")
REPLY_FORM = """Reply in exactly this form, with every tag present and nothing outside the tags:
<diagnosis>{diagnosis}</diagnosis>
<learnings>{learnings}; at most {learnings_words} words</learnings>
<rounds>a single number: {rounds}</rounds>
<shared>{shared}</shared>
{cards}"""
DIAGNOSIS_FORM = ("where this version of the team worked badly and where it worked well, each point with the tasks or "
                  "the statistics it rests on")
LEARNINGS_FORM = ("the updated learnings document: what has been seen to help and to hurt over the versions so far; it "
                  "replaces the old one, so keep what still holds")
RULES_ONLY_DIAGNOSIS_FORM = "where a team under these rules is likely to work badly, reasoned from the rules alone"
RULES_ONLY_LEARNINGS_FORM = "what you expect to help and to hurt"
SUPERVISOR_ASK = "Write your reply now, in the form given."
RULES_ONLY_ASK = "Set the team up now, in the form given."
CARD_TAG = re.compile(r"<card\s+seat\s*=\s*[\"']?\s*(?:agent\s*)?(\d+)\s*[\"']?\s*>(.*?)</card\s*>", re.S | re.I)
SHARED_PLACEHOLDER = "the shared notes"
CARD_PLACEHOLDER = "the card of agent {seat}"
ANY_CARD_PLACEHOLDER = re.compile(r"the card of agent \d+")
WORDS_SUFFIX = re.compile(r"[\W_]*at most \d+ words$")


def _choice(rounds) -> str:
    xs = [str(int(r)) for r in rounds]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


def rules_text(n: int = N_SEATS, rounds=ROUNDS_ALLOWED, hint: bool = True, comparison: bool = True) -> str:
    """The debate's rules as the supervisor reads them."""
    return RULES_TEXT.format(n=int(n), last=int(n) - 1, rounds=_choice(rounds), cut=CUT_SENTENCE,
                             stated=STATED_SENTENCE, asked=ASKED_SENTENCE, hint=HINT_SENTENCE if hint else "",
                             decision=DECISION_SENTENCE, comparison=COMPARISON_SENTENCE if comparison else "")


def _flat(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _cut(text: str, max_chars: int) -> str:
    text = _flat(text)
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + f" [cut: {len(text) - max_chars:,} more characters]"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _share(k: int, d: int):
    return round(k / d, 4) if d else None


def _answer_arg(args) -> str | None:
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict) or args.get("answer") is None:
        return None
    return str(args["answer"])


def _said(message: dict) -> str:
    c = message.get("content")
    if isinstance(c, list):
        c = " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return str(c).strip() if c and str(c).strip() else ""


def _turns(record: dict, j) -> list:
    entries = (record.get("transcripts") or {}).get(str(j)) or []
    out = []
    for e in entries:
        msgs = e.get("messages")
        if not isinstance(msgs, list):
            continue
        results = {m.get("tool_call_id"): str(m.get("content") or "") for m in msgs if m.get("role") == "tool"}
        text, answer, status = [], None, None
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            c = _said(m)
            if c:
                text.append(c)
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if fn.get("name", tc.get("name")) != SUBMIT_TOOL or status == "submitted":
                    continue
                res = results.get(tc.get("id"), "")
                answer = _answer_arg(fn.get("arguments", tc.get("arguments")))
                status = ("held" if res.startswith(HELD_PREFIX) else "submitted" if res.startswith(SUBMITTED_PREFIX)
                          else "refused")
        out.append({"step": int(e.get("step", len(out))), "text": "\n".join(text), "answer": answer, "status": status})
    return out


def _token_buckets(agent: dict) -> tuple:
    steps = [t.get("step") for t in agent.get("turns") or [] if (t.get("tools") or []) != ["forced_final"]]
    out, k, first = {}, 0, None
    for i, c in enumerate(agent.get("calls") or []):
        tok = int(c.get("completion_tokens") or 0)
        if i == 0 and not c.get("error"):
            first = tok
        tag = c.get("tag")
        if tag in ("turn", "deliberate") and k < len(steps):
            b = int(steps[k]) + 1
            k += 1
        elif tag == "final":
            b = "forced"
        else:
            b = "other"
        out[b] = out.get(b, 0) + tok
    return out, first


def episode(record: dict, example: dict, n: int) -> dict:
    """The label-free view of one debate record that the summaries and the statistics share."""
    agents = record.get("agents") or []
    rounds = int((record.get("method_state") or {}).get("rounds") or 0)
    hidden = list(example.get("hidden_information") or [])
    members, last_round = [], rounds
    for j, a in enumerate(agents):
        turns = {t["step"] + 1: t for t in _turns(record, a.get("id", j))}
        buckets, first = _token_buckets(a)
        final = (a.get("final") or {}).get("answer") if isinstance(a.get("final"), dict) else None
        forced = bool(a.get("final_forced"))
        answers = {r: norm(t["answer"]) for r, t in turns.items() if t["answer"] is not None and norm(t["answer"])}
        if forced and final is not None and norm(final):
            answers[max([rounds] + list(turns))] = norm(final)
        last_round = max([last_round] + list(turns))
        members.append({"id": a.get("id", j), "forced": forced, "final": None if final is None else str(final),
                        "turns": turns, "answers": answers, "tokens": buckets, "first_call_tokens": first})
    return {"id": str(record.get("example_id")), "rounds": rounds, "last_round": last_round, "members": members,
            "shared": list(example.get("shared_information") or []), "hidden": hidden,
            "total_tokens": sum(sum(m["tokens"].values()) for m in members)}


def _bucket_sums(eps: list) -> dict:
    by: dict = {}
    for ep in eps:
        for m in ep["members"]:
            for b, v in m["tokens"].items():
                by[b] = by.get(b, 0) + v
    last = max([ep["last_round"] for ep in eps] + [b for b in by if isinstance(b, int)])
    out = {r: by.get(r, 0) for r in range(1, last + 1)}
    out["forced"] = by.get("forced", 0)
    if by.get("other"):
        out["other"] = by["other"]
    return out


def episode_summary(record: dict, example: dict, n: int, max_chars: int = EPISODE_CHARS, tag: str | None = None) -> str:
    """One episode as the supervisor reads it: facts given, each member's text per round, submissions, tokens."""
    ep = episode(record, example, n)
    R = ep["rounds"]
    lines = [f"Task {tag or ep['id']} ({R} rounds): {_flat(example.get('description'))}",
             "  Options: " + " | ".join(_flat(o) for o in example.get("possible_answers") or []),
             "  Facts given to every member:"]
    lines += [f"    S{i + 1}. {_flat(s)}" for i, s in enumerate(ep["shared"])] or ["    (none)"]
    lines.append("  Facts given to one member only:")
    lines += [f"    P{h + 1}, agent {h % n}: {_flat(f)}" for h, f in enumerate(ep["hidden"])] or ["    (none)"]
    for r in range(1, ep["last_round"] + 1):
        lines.append(f"  Round {r}:")
        for m in ep["members"]:
            t = m["turns"].get(r)
            if t is None:
                done = any(x["status"] == "submitted" for q, x in m["turns"].items() if q < r)
                lines.append(f"    agent {m['id']}: no turn" + (" (it had submitted)." if done else "."))
                continue
            said = _flat(t["text"])
            body = ("wrote no text." if not said else "wrote: " + _cut(said, max_chars) if max_chars > 0
                    else f"wrote {len(said):,} characters.")
            ans = "an unreadable answer" if t["answer"] is None else f"\"{_flat(t['answer'])}\""
            did = {"held": f"; tried to submit {ans}, not accepted in round {r} and shown to the others as tentative",
                   "submitted": f"; submitted {ans}",
                   "refused": f"; tried to submit {ans}, refused by the host"}.get(t["status"],
                                                                                 "; did not submit" if r >= R else "")
            lines.append(f"    agent {m['id']} ({m['tokens'].get(r, 0):,} tokens{did}) {body}")
    for m in ep["members"]:
        if m["forced"]:
            ans = "no answer" if m["final"] is None else f"\"{_flat(m['final'])}\""
            lines.append(f"  agent {m['id']} had not submitted when the last round ended and was made to submit: {ans} "
                         f"({m['tokens'].get('forced', 0):,} tokens).")
    finals = [norm(m["final"]) or None for m in ep["members"]]
    shown = "; ".join(f"agent {m['id']} " + ("(none)" if m["final"] is None else f"\"{_flat(m['final'])}\"")
                      + (" (forced)" if m["forced"] else "") for m in ep["members"])
    given = [f for f in finals if f]
    top = max((given.count(f) for f in set(given)), default=0)
    lines.append(f"  Final answers: {shown}. {top} of {n} gave the same answer" + (" (all agreed)." if top == n else "."))
    tot = ep["total_tokens"]
    by = _bucket_sums([ep])
    parts = [f"round {b} {_pct(v / tot if tot else None)}" for b, v in by.items() if isinstance(b, int)]
    parts += [f"{name} {_pct(by[b] / tot if tot else None)}" for b, name in (("forced", "forced submissions"),
                                                                             ("other", "other calls")) if by.get(b)]
    lines.append(f"  Completion tokens: {tot:,} in the episode ({', '.join(parts)}); per member: "
                 + ", ".join(f"agent {m['id']} {sum(m['tokens'].values()):,}" for m in ep["members"]) + ".")
    return "\n".join(lines)


def batch_stats(records: list, rows: dict, n: int) -> dict:
    """Mechanical, label-free statistics of one team version over its batch of tasks."""
    eps = [episode(r, rows[str(r["example_id"])], n) for r in records]
    firsts_q = [first_turn_stats(r, rows[str(r["example_id"])], n) for r in records]
    R, E = eps[0]["rounds"], len(eps)
    by = _bucket_sums(eps)
    total = sum(by.values())
    last = max(ep["last_round"] for ep in eps)
    split = forced = members = 0
    firsts = []
    changed = [0, 0]
    for ep in eps:
        split += len({norm(m["final"]) or None for m in ep["members"]}) > 1
        for m in ep["members"]:
            members += 1
            forced += m["forced"]
            if m["first_call_tokens"] is not None:
                firsts.append(m["first_call_tokens"])
            if R - 1 in m["answers"] and R in m["answers"]:
                changed[1] += 1
                changed[0] += m["answers"][R - 1] != m["answers"][R]
    holders = [x for f in firsts_q for x in f["members"]]
    kept = [x for x in holders if x["left_out"] is None]
    left = Counter(x["left_out"] for x in holders if x["left_out"] is not None)
    with_private = [f for f in firsts_q if f["facts"]]
    on_table = sum(len(f["on_table"]) == f["facts"] for f in with_private)
    return {"episodes": E, "members": n, "rounds": R, "last_round": last,
            "completion_tokens_mean": round(sum(ep["total_tokens"] for ep in eps) / E, 1),
            "completion_token_share": {(f"round_{b}" if isinstance(b, int) else b): _share(v, total) for b, v in by.items()},
            "split_final_share": _share(split, E),
            "changed_answer_last_step": {"rounds": f"{R - 1}->{R}", "share": _share(*changed), "members": changed[1]},
            "forced_member_share": _share(forced, members),
            "first_round_members_counted": len(kept), "first_round_members_left_out": dict(sorted(left.items())),
            "first_round_own_facts_verbatim_share": _share(sum(bool(x["all"]) for x in kept), len(kept)),
            "all_private_verbatim_before_round_2_share": _share(on_table, len(with_private)),
            "first_call_completion_tokens_median": median(firsts) if firsts else None}


def stats_lines(stats: dict) -> str:
    sh = stats.get("completion_token_share") or {}
    names = {"forced": "forced submissions", "other": "other calls"}
    tok = ", ".join(f"{names.get(k, k.replace('_', ' '))} {_pct(v)}" for k, v in sh.items())
    ch = stats.get("changed_answer_last_step") or {}
    left = stats.get("first_round_members_left_out") or {}
    out = (f" ({sum(left.values())} more hold private facts and are not counted: "
           + ", ".join(f"{k} {v}" for k, v in left.items()) + ")") if left else ""
    return "\n".join([
        f"- tasks: {stats['episodes']}; members per task: {stats['members']}; rounds: {stats['rounds']}",
        f"- completion tokens per task (mean): {stats['completion_tokens_mean']:,}",
        f"- share of the completion tokens: {tok}",
        f"- tasks whose final answers were not all the same: {_pct(stats['split_final_share'])}",
        f"- members that changed the option they named from round {str(ch.get('rounds')).replace('->', ' to round ')}: "
        f"{_pct(ch.get('share'))} of the {ch.get('members')} that named one through the tool in both",
        f"- members made to submit because they had not when the last round ended: {_pct(stats['forced_member_share'])}",
        f"- members whose round-1 text held every private fact of their own word for word: "
        f"{_pct(stats['first_round_own_facts_verbatim_share'])} of {stats['first_round_members_counted']} members{out}",
        f"- tasks where every private fact was word for word in some member's round-1 text, on the table before "
        f"round 2: {_pct(stats['all_private_verbatim_before_round_2_share'])}"])


def stats_brief(stats: dict) -> str:
    ch = stats.get("changed_answer_last_step") or {}
    return (f"{stats['completion_tokens_mean']:,} completion tokens per task; final answers not all the same in "
            f"{_pct(stats['split_final_share'])} of tasks; {_pct(ch.get('share'))} changed the option they named "
            f"in the last step; {_pct(stats['forced_member_share'])} of members made to submit; own private facts "
            f"word for word in round 1 by {_pct(stats['first_round_own_facts_verbatim_share'])} of their holders; "
            f"every private fact word for word on the table before round 2 in "
            f"{_pct(stats['all_private_verbatim_before_round_2_share'])} of tasks")


def pick_episodes(example_ids, k: int = SUMMARY_EPISODES, salt: str = "") -> list:
    """The episodes the supervisor reads closely: a fixed hash order of the ids."""
    ids = sorted({str(i) for i in example_ids})
    return sorted(ids, key=lambda i: hashlib.sha256(f"{salt}/{i}".encode()).hexdigest())[:k]


def _system(job: str, given: str, diagnosis: str, learnings: str, rules: str, limits: dict) -> str:
    n = int(limits["n"])
    form = REPLY_FORM.format(diagnosis=diagnosis, learnings=learnings, learnings_words=limits["learnings_words"],
                             rounds=_choice(limits["rounds"]), shared=SHARED_PLACEHOLDER,
                             cards="\n".join(f'<card seat="{i}">{CARD_PLACEHOLDER.format(seat=i)}</card>'
                                             for i in range(n)))
    settings = SETTINGS.format(rounds=_choice(limits["rounds"]), shared_words=limits["shared_words"], last=n - 1,
                               card_words=limits["card_words"])
    return "\n\n".join([SUPERVISOR_ROLE.format(n=n) + job, rules, settings, given, WRITING, form])


def _team_view(team: dict) -> str:
    cards = team.get("cards") or []
    head = f"Current version (version {team.get('version')}): {team.get('rounds')} rounds."
    if not cards and not (team.get("shared") or "").strip():
        return head + "\nIt has no notes: its members read only their task."
    return "\n".join([head, "Shared notes:", (team.get("shared") or "").strip() or "(none)", "Cards:"]
                     + [f"agent {i}: {c}" for i, c in enumerate(cards)])


def _history_view(history: list) -> str:
    if not history:
        return "(none: the current version is the first)"
    out = []
    for e in history:
        line = f"Version {e['version']}: {e['rounds']} rounds" + ("" if e.get("notes", True) else ", no notes") + "."
        if e.get("stats"):
            line += "\n  " + stats_brief(e["stats"]) + "."
        if e.get("next_ok") is False:
            line += "\n  Your reply then could not be used, so the next version kept these notes and rounds."
        out.append(line)
    return "\n".join(out)


def supervisor_messages(rules: str, learnings: str, history: list, team: dict, summaries: list, stats: dict,
                        limits: dict = LIMITS) -> list:
    user = "\n\n".join(["Your learnings document so far:\n" + ((learnings or "").strip() or "(empty)"),
                        "Earlier versions:\n" + _history_view(history), _team_view(team),
                        f"Statistics of the current version over its {stats['episodes']} tasks:\n" + stats_lines(stats),
                        f"A closer account of {len(summaries)} of those tasks:\n\n" + "\n\n".join(summaries),
                        SUPERVISOR_ASK])
    return [{"role": "system", "content": _system(SUPERVISOR_JOB, SUPERVISOR_GIVEN, DIAGNOSIS_FORM, LEARNINGS_FORM,
                                                  rules, limits)},
            {"role": "user", "content": user}]


def rules_only_messages(rules: str, limits: dict = LIMITS) -> list:
    """The reference team: written from the rules alone, in the same reply form."""
    return [{"role": "system", "content": _system(RULES_ONLY_JOB, RULES_ONLY_GIVEN, RULES_ONLY_DIAGNOSIS_FORM,
                                                  RULES_ONLY_LEARNINGS_FORM, rules, limits)},
            {"role": "user", "content": RULES_ONLY_ASK}]


def clip_words(text: str, limit: int) -> tuple:
    lines = [" ".join(x.split()) for x in str(text or "").splitlines()]
    lines = [x for x in lines if x]
    total = sum(len(x.split()) for x in lines)
    if total <= limit:
        return "\n".join(lines), total
    out, left = [], int(limit)
    for x in lines:
        if left <= 0:
            break
        w = x.split()
        out.append(" ".join(w[:left]))
        left -= len(w)
    return "\n".join(out), total


def _tagged(tag: str, text: str) -> list:
    return re.findall(rf"<{tag}\s*>(.*?)</{tag}\s*>", text, re.S | re.I)


def _bare(text) -> str:
    return re.sub(r"^[\W_]+|[\W_]+$", "", _flat(text).lower())


def _is_form(text: str, forms: tuple) -> bool:
    got = _bare(text)
    return any(x in (_bare(f) for f in forms) for x in (got, _bare(WORDS_SUFFIX.sub("", got))))


def parse_supervisor(text: str, n: int, limits: dict, old_team: dict) -> tuple:
    """(team, report) read mechanically from a reply; on any fault the old team is kept and report['ok'] is False."""
    raw = str(text or "")
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    report: dict = {"ok": False, "reason": None, "rounds": None, "diagnosis": "", "learnings": "", "clipped": []}

    def fail(why: str):
        report["reason"] = why
        return json.loads(json.dumps(old_team)), report

    if re.search(r"<think\b", raw, re.I):
        return fail("a reasoning block is not closed")
    got, where = {}, raw
    for tag in ("diagnosis", "learnings", "rounds", "shared"):
        m = _tagged(tag, where)
        if not m:
            return fail(f"no <{tag}> tag")
        got[tag] = m[-1].strip()
        if tag == "diagnosis":
            where = re.sub(r"<diagnosis\s*>.*?</diagnosis\s*>", " ", raw, flags=re.S | re.I)
    seats = {int(s): body.strip() for s, body in CARD_TAG.findall(where)}
    nested = [t for t, body in [(t, got[t]) for t in ("learnings", "rounds", "shared")]
              + [(f"card {i}", b) for i, b in sorted(seats.items())] if FORM_TAG.search(body)]
    if nested:
        return fail(f"a tag of the form inside another ({', '.join(nested)})")
    missing = [i for i in range(n) if i not in seats]
    if missing or any(i >= n for i in seats):
        return fail(f"cards do not match the seats 0-{n - 1}")
    if not re.fullmatch(r"\d+", got["rounds"]) or int(got["rounds"]) not in tuple(limits["rounds"]):
        return fail(f"rounds {got['rounds'][:40]!r} is not one of {list(limits['rounds'])}")
    report["rounds"] = int(got["rounds"])
    report["diagnosis"] = got["diagnosis"]
    echoed = (["shared"] if _bare(got["shared"]) == SHARED_PLACEHOLDER else []) + [
        f"card {i}" for i in range(n) if ANY_CARD_PLACEHOLDER.fullmatch(_bare(seats[i]))] + (
        ["learnings"] if _is_form(got["learnings"], (LEARNINGS_FORM, RULES_ONLY_LEARNINGS_FORM)) else [])
    if echoed:
        return fail(f"the reply repeats the form's placeholder ({', '.join(echoed)})")

    def fit(name: str, body: str, limit: int) -> str:
        out, k = clip_words(unicodedata.normalize("NFC", body), limit)
        if k > limit:
            report["clipped"].append(name)
        return out

    shared = fit("shared", got["shared"], limits["shared_words"])
    report["learnings"] = fit("learnings", got["learnings"], limits["learnings_words"])
    cards = []
    for i in range(n):
        c = fit(f"card {i}", seats[i], limits["card_words"])
        if not c:
            return fail(f"the card of seat {i} is empty")
        cards.append(c)
    team = {"kind": KIND, "version": int(old_team.get("version", 0)) + 1, "rounds": report["rounds"], "shared": shared,
            "cards": cards}
    try:
        check_team(team, n)
    except ValueError as e:
        return fail(str(e))
    report["ok"] = True
    return team, report


def _tokens(text) -> list:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def text_leaks(text: str, rows: dict, min_words: int = LEAK_MIN_WORDS, where: str = "text") -> list:
    """Facts of the training tasks of which `text` holds a run of at least `min_words` consecutive words."""
    tok = _tokens(text)
    grams = {tuple(tok[i:i + min_words]) for i in range(len(tok) - min_words + 1)}
    hits = []
    for ex_id, row in rows.items():
        for kind in ("shared_information", "hidden_information"):
            for h, fact in enumerate(row.get(kind) or []):
                ft = _tokens(fact)
                if any(tuple(ft[i:i + min_words]) in grams for i in range(len(ft) - min_words + 1)):
                    hits.append({"where": where, "example": str(ex_id), "fact": f"{kind}[{h}]"})
    return hits


def leak_check(team: dict, rows: dict) -> list:
    hits = text_leaks(team.get("shared") or "", rows, where="shared")
    for i, c in enumerate(team.get("cards") or []):
        hits += text_leaks(c, rows, where=f"card {i}")
    return hits


def load_rows(dirs: list) -> dict:
    """Task inputs only (no answer fields) of every training task, by example id."""
    out = {}
    for d in dirs:
        for row in load_benchmark(d):
            out[f"hb_{int(row['id']):02d}"] = {k: row[k] for k in INPUT_FIELDS if k in row}
    return out


def blind_record(path: str) -> dict:
    """A record with every scoring field dropped while it is parsed."""
    def hook(pairs):
        return {k: v for k, v in pairs if k not in SCORE_KEYS}

    with gzip.open(path, "rt") as f:
        return json.load(f, object_pairs_hook=hook)


async def ask(llm, messages: list, route: str, max_tokens: int) -> str:
    c = await llm.chat(messages, tools=None, max_tokens=max_tokens, temperature=1.0, enable_thinking=True, tag="supervisor",
                       route_key=route)
    if c.error:
        raise SystemExit(f"the supervisor call failed: {c.error}")
    return c.content or ""


def write_team(path: Path, team: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(team, indent=1) + "\n")


def dev_run(a, team_path: Path, rounds: int, version_dir: Path, ids: list) -> list:
    """Run the 3-step debate with the team file on the version's tasks and return its records, scores dropped."""
    from rci.run import main as run_main

    root = version_dir / "records"
    kw = {**json.loads(a.protocol_kwargs), "rounds": rounds}
    if json.loads(team_path.read_text()).get("cards"):
        kw["team_file"] = str(team_path)
    task_kw = {"root": a.train.split(",")[0], **json.loads(a.task_kwargs)}
    argv = ["--task", "hiddenbench", "--task-kwargs", json.dumps(task_kw), "--protocol", "debate", "--protocol-kwargs",
            json.dumps(kw), "--n-agents", str(N_SEATS), "--seeds", str(a.seed), "--example-ids", ",".join(ids),
            "--base-urls", a.base_urls, "--model", a.model, "--out", str(root), "--max-steps", str(rounds),
            "--save-transcripts", "--concurrency", str(a.concurrency)]
    if run_main(argv):
        raise SystemExit("the version's run failed")
    files = sorted(glob.glob(str(root / "*" / f"N{N_SEATS}" / f"s{a.seed}" / "*.json.gz")))
    return [blind_record(f) for f in files]


async def chain(a) -> None:
    from rci.agents.llm import LLMClient

    K = Path(a.out)
    K.mkdir(parents=True, exist_ok=True)
    rows = load_rows(a.train.split(","))
    ids = pick_episodes(load_rows(a.train.split(",")[:1]), a.dev_tasks, salt=f"dev/{a.chain}")
    rules = rules_text()
    llm = LLMClient(a.base_urls.split(","), a.model, max_concurrency=8)
    if a.rules_only:
        path = K / "rules_only" / "team.json"
        if not path.exists():
            base = {"kind": KIND, "version": 0, "rounds": max(ROUNDS_ALLOWED), "shared": "", "cards": []}
            for attempt in range(2):
                text = await ask(llm, rules_only_messages(rules), f"rules_only/{a.chain}/a{attempt}", a.max_tokens)
                team, rep = parse_supervisor(text, N_SEATS, LIMITS, base)
                if rep["ok"] and not leak_check(team, rows):
                    write_team(path, team)
                    break
        return
    history, learnings = [], ""
    team = {"kind": KIND, "version": 0, "rounds": a.rounds, "shared": "", "cards": []}
    for k in range(a.versions + 1):
        vdir = K / f"v{k}"
        team_path = vdir / "team.json"
        if team_path.exists():
            team = json.loads(team_path.read_text())
        else:
            write_team(team_path, team)
        recs = await asyncio.to_thread(dev_run, a, team_path, int(team["rounds"]), vdir, ids)
        stats = batch_stats(recs, rows, N_SEATS)
        (vdir / "stats.json").write_text(json.dumps(stats, indent=1))
        if k == a.versions:
            break
        nxt = K / f"v{k + 1}" / "team.json"
        if nxt.exists():
            history.append({"version": k, "rounds": team["rounds"], "notes": bool(team.get("cards")), "stats": stats})
            learnings = (K / f"v{k + 1}" / "learnings.md").read_text() if (K / f"v{k + 1}" / "learnings.md").exists() \
                else learnings
            team = json.loads(nxt.read_text())
            continue
        by_id = {str(r["example_id"]): r for r in recs}
        picked = pick_episodes(by_id, SUMMARY_EPISODES, salt=f"summaries/{a.chain}/v{k}")
        summaries = [episode_summary(by_id[q], rows[q], N_SEATS) for q in picked]
        msgs = supervisor_messages(rules, learnings, history, team, summaries, stats)
        ok, new, rep = False, None, None
        for attempt in range(2):
            text = await ask(llm, msgs, f"supervisor/{a.chain}/v{k + 1}/a{attempt}", a.max_tokens)
            new, rep = parse_supervisor(text, N_SEATS, LIMITS, team)
            if rep["ok"] and not leak_check(new, rows) and not text_leaks(rep["learnings"], rows) \
                    and not SECTION.search(rep["learnings"] + "\n"):
                ok = True
                break
        history.append({"version": k, "rounds": team["rounds"], "notes": bool(team.get("cards")), "stats": stats,
                        "next_ok": ok})
        if ok:
            learnings = rep["learnings"]
        else:
            new = {**json.loads(json.dumps(team)), "version": k + 1}
        write_team(nxt, new)
        (nxt.parent / "learnings.md").write_text(learnings + "\n")
        with open(K / "history.jsonl", "a") as f:
            f.write(json.dumps({"version": k, "next_ok": ok, "reason": None if ok else rep.get("reason"),
                                "stats": stats}) + "\n")
        team = new


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evolve a debate team with a supervisor that reads the team's unlabeled "
                                             "records on training tasks and rewrites its rounds, shared notes and "
                                             "cards; or write the reference team from the rules alone.")
    ap.add_argument("--train", required=True, help="comma list of training task directories holding benchmark.json; "
                                                   "the version runs draw their tasks from the first")
    ap.add_argument("--chain", default="c0", help="name of the chain (fixes the task and episode draws)")
    ap.add_argument("--versions", type=int, default=2, help="supervisor steps; versions 0..K are run")
    ap.add_argument("--rounds", type=int, default=3, choices=list(ROUNDS_ALLOWED), help="rounds of version 0")
    ap.add_argument("--dev-tasks", type=int, default=DEV_TASKS)
    ap.add_argument("--task-kwargs", default='{"profile": "hidden", "hint": true}')
    ap.add_argument("--protocol-kwargs", default="{}", help="further debate kwargs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--rules-only", action="store_true", help="write only the team set up from the rules alone")
    ap.add_argument("--base-urls", default=os.environ.get("RCI_BASE_URLS", ""))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="chain directory: v<k>/team.json, stats, learnings, history")
    a = ap.parse_args(argv)
    if not a.base_urls:
        ap.error("--base-urls (or RCI_BASE_URLS) is required")
    asyncio.run(chain(a))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
