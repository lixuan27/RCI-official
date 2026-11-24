"""Self-written prompts and role cards a blackboard member reads before a task."""
from __future__ import annotations

import json
from pathlib import Path

CARD_KINDS = ("roles",)


def notes_block(kind: str, text: str, seat: int) -> str:
    """The block appended to a member's prompt: shared notes, its own card, or the role set for its seat."""
    if not text:
        return ""
    if kind == "roles":
        return f"\n\nInstructions for your seat, Participant {seat} (set before the task):\n{text}"
    return f"\n\nNotes from earlier groups under the same rules (participants keep their numbers):\n{text}"


def load_notes(path) -> dict:
    """A prompt file: {"kind": "rules"|"shared", "notes": str} or {"kind": "roles", "cards": [str per member]}."""
    d = json.loads(Path(path).read_text())
    if not isinstance(d, dict) or "kind" not in d:
        raise ValueError(f"{path}: a prompt file holds a JSON object with a `kind`")
    if d["kind"] in CARD_KINDS:
        if not isinstance(d.get("cards"), list):
            raise ValueError(f"{path}: kind {d['kind']!r} needs a list of `cards`")
    elif not isinstance(d.get("notes"), str):
        raise ValueError(f"{path}: kind {d['kind']!r} needs `notes` text")
    return d
