"""First turns of debate records: visible text, tool calls, sealed loading and the guard that no label is read."""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

from rci.tasks.hiddenbench.facts import keys_of

FORBIDDEN_KEYS = frozenset({"correct_answer", "full_answer", "full_agree", "score", "finals", "final", "preds",
                            "per_agent"})
FORBIDDEN_FILES = frozenset({"writer_key.json"})
SUBMIT_TOOL = "submit_answer"
FORCED_TURN = ["forced_final"]
CALL_TEXT = re.compile(r"<tool_call>|<function=")
THINK = re.compile(r"<think>.*?</think>", re.S)
PROTOCOL_DIRS = {"a": "debate_r2_forward_facts", "f": "debate_r3"}
QUOTES = re.compile("[\u2018\u2019\u201c\u201d'\"`*_]")


class TruthRead(AssertionError):
    """A label field or the writer key was read while building label-free rows."""


class Sealed(dict):
    """A record or benchmark row whose label fields raise TruthRead when read and are hidden from iteration."""

    def _check(self, k):
        if k in FORBIDDEN_KEYS:
            raise TruthRead(f"label-free code never reads {k!r}")

    def __getitem__(self, k):
        self._check(k)
        return seal(dict.__getitem__(self, k))

    def get(self, k, default=None):
        self._check(k)
        return seal(dict.get(self, k, default))

    def __iter__(self):
        return (k for k in dict.__iter__(self) if k not in FORBIDDEN_KEYS)

    def keys(self):
        return list(iter(self))

    def items(self):
        return [(k, self[k]) for k in self]

    def values(self):
        return [self[k] for k in self]


def seal(x):
    if isinstance(x, Sealed):
        return x
    if isinstance(x, dict):
        return Sealed(x)
    if isinstance(x, list):
        return [seal(v) for v in x]
    return x


_GUARD = {"armed": False, "hooked": False}


def _audit(event: str, args) -> None:
    if _GUARD["armed"] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if os.path.basename(os.fsdecode(args[0])) in FORBIDDEN_FILES:
            raise TruthRead(f"label-free code never opens {os.fsdecode(args[0])}")


@contextmanager
def no_truth():
    """While active, opening the writer key raises TruthRead (a process audit hook)."""
    if not _GUARD["hooked"]:
        sys.addaudithook(_audit)
        _GUARD["hooked"] = True
    _GUARD["armed"] = True
    try:
        yield
    finally:
        _GUARD["armed"] = False


def guard_report() -> dict:
    return {"forbidden_keys": sorted(FORBIDDEN_KEYS), "forbidden_files": sorted(FORBIDDEN_FILES),
            "records_and_rows_sealed": True, "writer_key_hook": _GUARD["hooked"]}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def ids_hash(ids: list) -> str:
    return sha(json.dumps(list(ids)))


def example_id(row) -> str:
    return f"hb_{int(row['id']):02d}"


def load_bench(dirs: list) -> tuple:
    """Every question of the given task directories by example id, sealed, and each file's hash."""
    bench, info = {}, []
    for d in dirs:
        p = Path(d) / "benchmark.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run scripts/data/prepare_splitbench.sh (see README, Data)")
        raw = p.read_bytes()
        rows = [seal(x) for x in json.loads(raw)]
        for row in rows:
            q = example_id(row)
            if q in bench:
                raise SystemExit(f"{q} is in two task sets ({d}): the sets must not overlap")
            bench[q] = row
        info.append({"name": Path(d).name, "questions": len(rows), "sha256": hashlib.sha256(raw).hexdigest()})
    return bench, info


def text_of(msg: dict) -> str:
    """A transcript message's text without its reasoning."""
    c = msg.get("content")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return THINK.sub("", str(c or ""))


def visible(text: str) -> str:
    """A raw generation's text after its reasoning."""
    return THINK.sub("", (text or "").split("</think>")[-1])


def content(text: str) -> str:
    """The visible text before any tool call: what the harness keeps as the message."""
    vis = visible(text)
    m = CALL_TEXT.search(vis)
    return vis[:m.start()] if m else vis


def has_tool_call(text: str) -> bool:
    return bool(CALL_TEXT.search(visible(text)))


def calls_of(msgs: list) -> list:
    """Tool calls of a turn's generations by name; '(unparsed)' for a call left in the text."""
    out = []
    for m in msgs:
        out += [((tc.get("function") or {}).get("name") or "?") for tc in (m.get("tool_calls") or [])]
        if CALL_TEXT.search(str(m.get("content") or "")):
            out.append("(unparsed)")
    return out


def call_reason(calls: list) -> str:
    return "first turn submits an answer" if SUBMIT_TOOL in calls or "(unparsed)" in calls else \
        "first turn makes a tool call"


def special_ids(tokenizer_path: str) -> dict:
    """Token ids of the reasoning end and the tool-call markers in a model's tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    out = {k: tok.convert_tokens_to_ids(v) for k, v in
           (("think_end", "</think>"), ("call_open", "<tool_call>"), ("call_close", "</tool_call>"))}
    if any(v is None or v == tok.unk_token_id for v in out.values()):
        raise SystemExit(f"{tokenizer_path}: the tokenizer has no reasoning or tool-call tokens")
    return out


def target_len(ids: list, start: int, n: int, called: bool, submit: str, call_id: int | None) -> int | None:
    """Loss length of a first span: n, or with submit='cut' the tokens before the first tool-call token."""
    if submit != "cut" or not called:
        return n
    span = ids[start:start + n]
    k = span.index(call_id) if call_id in span else -1
    return k if k > 0 else None


def mine_of(row, n: int, seat: int) -> list:
    """The private facts held by a seat: fact h goes to seat h mod n."""
    return [h for h in range(len(keys_of(row))) if h % n == seat]


def norm(s: str) -> str:
    """Case, whitespace and quotes folded."""
    return re.sub(r"\s+", " ", QUOTES.sub("", s.lower())).strip()


def facts_of(row) -> list:
    return [x if isinstance(x, str) else (x.get("text") or x.get("fact") or json.dumps(x))
            for x in row["hidden_information"]]


def real_turns(agent: dict) -> list:
    return [t for t in agent.get("turns") or [] if isinstance(t, dict) and list(t.get("tools") or []) != FORCED_TURN]


def first_messages(record: dict, seat: int) -> list:
    """The assistant messages of a seat's first turn, from the record's transcripts."""
    log = (record.get("transcripts") or {}).get(str(seat)) or []
    if not log:
        return []
    return [m for m in (log[0].get("messages") or []) if m.get("role") == "assistant"]


def first_turn_stats(record: dict, example, n: int = 4) -> dict:
    """Round one of a debate record: each member's share of own facts written word for word, and facts on the table."""
    agents = record.get("agents") or []
    facts = [norm(f) for f in facts_of(example)]
    members, on_table = [], set()
    for j, a in enumerate(agents[:n]):
        said = first_messages(record, a.get("id", j))
        for m in said:
            body = norm(content(text_of(m)))
            on_table |= {h for h, f in enumerate(facts) if f and f in body}
        mine = [h for h in range(len(facts)) if h % n == j]
        if not mine:
            continue
        calls = [c for c in a.get("calls") or [] if isinstance(c, dict)]
        if a.get("failing") or not calls or calls[0].get("error") or not said:
            members.append({"left_out": "failed"})
        elif len(said) != 1:
            members.append({"left_out": "several generations"})
        else:
            body = norm(content(text_of(said[0])))
            hit = sum(facts[h] in body for h in mine)
            members.append({"left_out": None, "share": hit / len(mine), "all": hit == len(mine)})
    return {"members": members, "facts": len(facts), "on_table": on_table}


def harvest_records(runs: list, proto_dir: str, n: int = 4, seeds: set | None = None):
    """(path, seed, sealed record) of every record of one protocol directory under each harvest root."""
    for d in runs:
        for p in sorted(glob.glob(os.path.join(d, proto_dir, f"N{n}", "s*", "*.json.gz"))):
            seed = int(Path(p).parent.name[1:])
            if seeds is not None and seed not in seeds:
                continue
            with gzip.open(p, "rt") as f:
                yield p, seed, seal(json.load(f))


def protocol_dirs(spec: str | None) -> dict:
    """'a=DIR,f=DIR' -> {'a': DIR, 'f': DIR}, defaulting to PROTOCOL_DIRS."""
    out = dict(PROTOCOL_DIRS)
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def seeds_of(s: str | None) -> set | None:
    return {int(x) for x in s.split(",") if x} if s else None
