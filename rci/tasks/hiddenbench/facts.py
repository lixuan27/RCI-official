"""Which member holds which private fact: fact keys, matching messages to facts, and fact views of a text."""
from __future__ import annotations

import re

STOP = set("""a an the and or but if of to in on at by for with from into onto over under about after before during while
this that these those it its is are was were be been being has have had do does did not no yes can could will would should
may might must shall also only just than then there their they them he she his her him we our you your i me my one two
three some any all each other more most very much many such same own so as up down out off again further once here when
where why how what which who whom whose because until against between through above below both few nor too s t don now""".split())


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in STOP}


def keys_of(row: dict) -> list[set[str]]:
    """The matching key of each private fact of a benchmark row."""
    common = words(" ".join(row["shared_information"] + row["possible_answers"] + [row["description"]]))
    out = []
    for h in row["hidden_information"]:
        w = words(h)
        d = w - common
        out.append(d if len(d) >= 2 else w)
    return out


def holds(msg_words: set[str], key: set[str], thr: float = 0.5) -> bool:
    return bool(key) and len(key & msg_words) / len(key) >= thr


def holders(row: dict, text: str, n: int, thr: float = 0.5) -> tuple[list[int], bool]:
    w = words(text)
    who = sorted({h % n for h, key in enumerate(keys_of(row)) if holds(w, key, thr)})
    return who, any(holds(w, words(s), thr) for s in row["shared_information"])


def key_collisions(row: dict, thr: float = 0.5) -> list[tuple[int, int]]:
    keys, hidden = keys_of(row), row["hidden_information"]
    return [(i, j) for i in range(len(keys)) for j in range(len(hidden)) if i != j and holds(words(hidden[j]), keys[i], thr)]


def multi_fact_errors(row: dict, sizes=(2, 3), thr: float = 0.5) -> list[tuple[int, ...]]:
    from itertools import combinations
    keys, hidden = keys_of(row), row["hidden_information"]
    out = []
    for k in sizes:
        for s in combinations(range(len(hidden)), k):
            w = words("\n".join(hidden[i] for i in s))
            if {h for h, key in enumerate(keys) if holds(w, key, thr)} != set(s):
                out.append(s)
    return out


def message_text(m: dict) -> str:
    """A store message as one string: what was done, what it produced, and why."""
    return " | ".join(str(m.get(k) or "") for k in ("sub_task", "artifact", "rationale"))


def members(rec: dict, row: dict, thr: float = 0.5) -> list[dict]:
    keys = keys_of(row)
    n = len(rec["agents"])
    ms = rec["method_state"]
    msgs = {m["id"]: m for m in ms.get("messages", [])}
    mw = {i: words(message_text(m)) for i, m in msgs.items()}
    last = {a["id"]: max((t["step"] for t in a.get("turns") or []), default=0) for a in rec["agents"]}
    shown: dict[int, dict[int, int]] = {a["id"]: {} for a in rec["agents"]}
    for e in ms.get("events", []):
        if e["op"] == "search":
            ids = [r["id"] for r in e.get("results") or []]
        elif e["op"] == "visit":
            ids = [int(x) for x in e.get("ids") or [] if str(x).isdigit()]
        elif e["op"] == "push" and e.get("view") == "full":
            ids = list(e.get("ids") or [])
        else:
            continue
        j = e.get("reader")
        if j in shown and (e["step"] < last[j] or (e["op"] == "push" and e["step"] <= last[j])):
            for i in ids:
                if i in msgs and msgs[i]["author"] != j:
                    shown[j].setdefault(i, e["step"])
    out = []
    for a in rec["agents"]:
        j = a["id"]
        has = {h for h in range(len(keys)) if h % n == j}
        done_at = -1 if len(has) == len(keys) else None
        for i, st in sorted(shown[j].items(), key=lambda x: (x[1], x[0])):
            has |= {h for h, k in enumerate(keys) if holds(mw[i], k, thr)}
            if done_at is None and len(has) == len(keys):
                done_at = st
        written = sorted((i for i, m in msgs.items() if m["author"] == j and m["step"] < last[j]), key=lambda i: (msgs[i]["step"], i))
        out.append({"agent": j, "shown": sorted(shown[j], key=lambda i: (shown[j][i], i)), "shown_step": shown[j],
                    "written": written, "has": has, "complete": len(has) == len(keys), "completed_at": done_at,
                    "last": last[j]})
    return out


SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+|\s+\|\s+")


def fact_sentences(text: str, row: dict, thr: float = 0.5) -> str:
    keys = keys_of(row) + [words(x) for x in row["shared_information"]]
    keep = [x.strip() for x in SENTENCE.split(text) if x.strip() and any(holds(words(x), k, thr) for k in keys)]
    return " ".join(keep)


VERDICT = re.compile(r"\b(i think|i believe|i recommend|we should|should (choose|pick|go with)|best (option|choice)|"
                     r"my (final )?(answer|choice|vote|pick|recommendation)|the answer is|i (lean|favou?r|pick|choose|select)|"
                     r"i('m| am) going (for|with)|recommend(ation)?|i('d| would) (choose|pick|go with))\b", re.I)


def verdict_of(text: str, row: dict) -> str | None:
    corpus = " ".join([row.get("description", ""), *row["shared_information"], *row["hidden_information"]]).lower()
    opts = [re.compile(rf"\b{re.escape(o.lower())}\b") for o in row["possible_answers"]]
    for s in SENTENCE.split(text):
        low = s.lower()
        if any(o.search(low) for o in opts) and any(m.group(0) not in corpus for m in VERDICT.finditer(low)):
            return s.strip()
    return None


def answer_view(text: str, row: dict) -> str | None:
    v = verdict_of(text, row)
    if v is not None:
        named = [o for o in row["possible_answers"] if re.search(rf"\b{re.escape(o.lower())}\b", v.lower())]
        if len(named) == 1:
            return f"It recommends: {named[0]}."
    keys = keys_of(row) + [words(x) for x in row["shared_information"]]
    rest = [x.strip() for x in SENTENCE.split(text) if x.strip() and not any(holds(words(x), k) for k in keys)]
    return " ".join(rest) or None


def written_facts(rec: dict, row: dict, thr: float = 0.5) -> list[bool]:
    """Whether each private fact reached the store at all."""
    mw = [words(message_text(m)) for m in rec["method_state"].get("messages", [])]
    return [any(holds(w, k, thr) for w in mw) for k in keys_of(row)]
