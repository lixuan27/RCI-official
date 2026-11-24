"""Supervisor-written team files for the debate: format, checks, loading, and the block each member reads."""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

N_SEATS = 4
ROUNDS_ALLOWED = (2, 3)
SHARED_WORDS = 150
CARD_WORDS = 120
LEARNINGS_WORDS = 300
SUMMARY_EPISODES = 12
LEAK_MIN_WORDS = 6
LIMITS = {"n": N_SEATS, "rounds": ROUNDS_ALLOWED, "shared_words": SHARED_WORDS, "card_words": CARD_WORDS,
          "learnings_words": LEARNINGS_WORDS}
MESSAGE_CHARS = 600

KIND = "team"
SUBMIT_TOOL = "submit_answer"
HELD_PREFIX = "Not submitted"
SUBMITTED_PREFIX = "Answer submitted"
SPECIAL = re.compile(r"<\|[^|>\n]*\|>")
FORM_TAG = re.compile(r"</?\s*(?:diagnosis|learnings|rounds|shared|card|think)\b[^<>]*>", re.I)


TEAM_LEAD = ("Team notes, written before this task. You are agent {seat} of your team; the other members' work reaches "
             "you under their agent numbers. Rounds are counted from 1: your first turn is round 1.")
SHARED_HEAD = "For every member:"
CARD_HEAD = "Your card, agent {seat}:"


def team_block(shared: str, card: str, seat: int) -> str:
    if not isinstance(shared, str) or not isinstance(card, str):
        raise ValueError("team_block: shared and card are strings")
    shared, card = shared.strip(), card.strip()
    if not card:
        raise ValueError(f"team_block: the card of seat {seat} is empty")
    out = "\n\n" + TEAM_LEAD.format(seat=int(seat))
    if shared:
        out += "\n\n" + SHARED_HEAD + "\n" + shared
    return out + "\n\n" + CARD_HEAD.format(seat=int(seat)) + "\n" + card


def strip_team_block(system_text: str, block: str) -> str:
    if not block:
        raise ValueError("strip_team_block: empty block")
    k = system_text.count(block)
    if k != 1:
        raise ValueError("strip_team_block: team block not found" if k == 0 else
                         f"strip_team_block: team block repeated ({k} times)")
    return system_text.replace(block, "", 1)


def notes_fault(text: str) -> str | None:
    m = SPECIAL.search(text)
    if m:
        return f"hold a chat-template marker ({m.group(0)!r})"
    m = FORM_TAG.search(text)
    if m:
        return f"hold a tag of the supervisor's reply form ({m.group(0)!r})"
    if not unicodedata.is_normalized("NFC", text):
        return "are not in Unicode NFC form"
    return None


def check_team(team, n: int | None = None, allow_no_notes: bool = False) -> dict:
    if not isinstance(team, dict):
        raise ValueError("a team file holds a JSON object")
    missing = [k for k in ("kind", "version", "rounds", "shared", "cards") if k not in team]
    if missing:
        raise ValueError(f"team file without {missing}"
                         + (": a team file carries its rounds, 2 or 3" if "rounds" in missing else ""))
    if team["kind"] != KIND:
        raise ValueError(f"team file of kind {team['kind']!r}, not {KIND!r}")
    if isinstance(team["version"], bool) or not isinstance(team["version"], int) or team["version"] < 0:
        raise ValueError(f"team version {team['version']!r} is not a non-negative integer")
    if isinstance(team["rounds"], bool) or not isinstance(team["rounds"], int) or team["rounds"] not in ROUNDS_ALLOWED:
        raise ValueError(f"team rounds {team['rounds']!r} is not one of {list(ROUNDS_ALLOWED)}")
    if not isinstance(team["shared"], str):
        raise ValueError("team shared notes are not a string")
    cards = team["cards"]
    if not isinstance(cards, list):
        raise ValueError("team cards are not a list")
    if allow_no_notes and not cards and not team["shared"].strip():
        return team
    for i, c in enumerate(cards):
        if not isinstance(c, str) or not c.strip():
            raise ValueError(f"team card {i} is not a non-empty string")
    for where, text in [("shared notes", team["shared"])] + [(f"card {i}", c) for i, c in enumerate(cards)]:
        why = notes_fault(text)
        if why:
            raise ValueError(f"team {where} {why}")
    if not cards:
        raise ValueError("team without cards")
    if n is not None and len(cards) != n:
        raise ValueError(f"{len(cards)} cards for {n} members")
    return team


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_team(path, n: int | None = None) -> tuple[dict, str]:
    """A team file read once: (the checked team, the sha256 of the file's bytes)."""
    data = Path(path).read_bytes()
    try:
        team = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"team file {path} is not JSON: {e}") from None
    try:
        return check_team(team, n), sha256_of(data)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None


_LOADED: dict = {}


def load_team_once(path, n: int | None = None) -> tuple[dict, str]:
    key = os.path.abspath(os.fspath(path))
    hit = _LOADED.get(key)
    try:
        st = os.stat(key)
        sig = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    except OSError:
        if hit is None:
            raise
        sig = hit[0]
    if hit is None or hit[0] != sig:
        team, sha = load_team(path)
        hit = _LOADED[key] = (sig, team, sha)
    if n is not None and len(hit[1]["cards"]) != n:
        raise ValueError(f"{path}: {len(hit[1]['cards'])} cards for {n} members")
    return hit[1], hit[2]


def notes_check(record: dict, team: dict | None, sha256: str | None = None) -> list[str]:
    ms = record.get("method_state") or {}
    has = [k for k in ("notes_file", "notes_sha256", "notes_version", "notes_blocks", "notes_in_prompt") if k in ms]
    if team is None:
        return [f"the record ran under notes ({', '.join(has)})"] if has else []
    if len(has) != 5:
        return ["the record ran without notes" if not has else f"the record's notes fields are incomplete ({', '.join(has)})"]
    out = []
    if sha256 is not None and ms["notes_sha256"] != sha256:
        out.append(f"notes sha256 {str(ms['notes_sha256'])[:12]} is not {sha256[:12]}")
    if ms["notes_version"] != team.get("version"):
        out.append(f"notes version {ms['notes_version']} is not {team.get('version')}")
    cards = team.get("cards") or []
    want = [team_block(team.get("shared") or "", c, i) for i, c in enumerate(cards)]
    if ms["notes_blocks"] != want:
        out.append("the members' blocks are not the team's")
    if len(record.get("agents") or []) != len(cards):
        out.append(f"{len(record.get('agents') or [])} members for {len(cards)} cards")
    if not all(x is True for x in ms["notes_in_prompt"]) or len(ms["notes_in_prompt"]) != len(cards):
        out.append("a member's system prompt did not hold its block exactly once")
    if team.get("rounds") is None:
        out.append("the team states no rounds")
    elif ms.get("rounds") != team["rounds"]:
        out.append(f"the debate ran {ms.get('rounds')} rounds, the team sets {team['rounds']}")
    return out
