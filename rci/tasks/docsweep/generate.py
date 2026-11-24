"""DocSweep generator: synthetic corpora whose answer is spread over many documents."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import time
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from rci import paths
from rci.tasks.docsweep import normalise as N

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
DOMAINS = N.DOMAINS
FORMATS = ("report", "log", "email")
DATE_STYLES = ("iso", "dmy", "mdy")
FAMILIES = ("agg", "join")
K_DEFAULT = 24
TARGET_DOC_TOKENS = 1400
DOC_CAP = 1800
CORPUS_CAP = 34000
ORACLE_CAP = 36864
ORACLE_OVERHEAD = 2000
MAX_ATTEMPTS = 30
OVERFILL = 1.25
WORDS_P = 0.12
YEAR = 2024
DISTRACTORS = (6, 10)
NEAR = (1, 3)
DECOY_MENTIONS = (5, 7)
UNIT_FORMS = {
    "L": (1, ("litres", "L"), "L"),
    "m3": (1000, ("cubic metres", "m³"), "m³"),
    "m": (1, ("metres", "m"), "m"),
    "km": (1000, ("kilometres", "km"), "km"),
    "kg": (1, ("kilograms", "kg"), "kg"),
    "t": (1000, ("tonnes", "t"), "t"),
    "m2": (1, ("square metres", "m²"), "m²"),
    "ha": (10000, ("hectares", "ha"), "ha"),
}


class Regenerate(Exception):
    pass


class ParaphraseError(RuntimeError):
    """The paraphrase service failed (transport, HTTP status or a malformed reply) after its retries."""


def render_doc(d: int, k: int, text: str) -> str:
    return f"Document {d} of {k}\n{text}"


def doc_budget(k: int) -> int:
    return min(DOC_CAP, CORPUS_CAP // k)


def answer_range(k: int) -> tuple[int, int]:
    lo = max(2, math.ceil(4 * k / 24))
    return lo, max(lo + 1, math.floor(10 * k / 24))


def common() -> dict:
    return json.loads((TEMPLATES / "common.json").read_text(encoding="utf-8"))


def fmt_qty(v: int, unit: str, rng: random.Random, short: bool) -> str:
    factor, forms, sym = UNIT_FORMS[unit]
    name = sym if short else rng.choice(forms)
    if factor == 1:
        if not short and v % 100 == 0 and 100 <= v <= 9900 and rng.random() < WORDS_P:
            num = N.number_words(v)
        else:
            num = f"{v:,}"
    else:
        s = format((Decimal(v) / Decimal(factor)).normalize(), "f")
        whole, _, frac = s.partition(".")
        num = f"{int(whole):,}" + (f".{frac}" if frac else "")
    return f"{num} {name}"


def fmt_date(d: date, style: str) -> str:
    month = N.MONTHS[d.month - 1]
    if style == "iso":
        return d.isoformat()
    if style == "dmy":
        return f"{d.day} {month} {d.year}"
    return f"{month} {d.day}, {d.year}"


def fmt_count(n: int, rng: random.Random, short: bool) -> str:
    return str(n) if short or rng.random() < 0.5 else N.number_words(n)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _rand_date(rng: random.Random, lo: date = date(YEAR, 1, 10), hi: date = date(YEAR, 12, 15)) -> date:
    return lo + timedelta(days=rng.randrange((hi - lo).days + 1))


def _draw(rng: random.Random, lo: int, hi: int, step: int) -> int:
    a, b = -(-lo // step), hi // step
    if a > b:
        raise Regenerate(f"empty range [{lo}, {hi}] step {step}")
    return rng.randint(a, b) * step


def _people(rng: random.Random, n: int, used: set) -> list[tuple[str, str]]:
    c = common()
    out = []
    while len(out) < n:
        p = (rng.choice(c["first_names"]), rng.choice(c["last_names"]))
        if p not in used:
            used.add(p)
            out.append(p)
    return out


def _region(rng: random.Random) -> str:
    c = common()
    return f"{rng.choice(c['region_adj'])} {rng.choice(c['region_names'])} {rng.choice(c['region_kinds'])}"


def _entities(T: dict, types: list[str], rng: random.Random) -> list[dict]:
    used1: set = set()
    people: set = set()
    codes = rng.sample(range(1000, 10000), len(types))
    out = []
    for i, typ in enumerate(types):
        spec = T["types"][typ]
        s1 = rng.choice([w for w in spec["stems1"] if w not in used1])
        used1.add(s1)
        stem = f"{s1} {rng.choice(spec['stems2'])}"
        out.append({"type": typ, "name": f"{stem} {spec['suffix']}", "stem": stem,
                    "aliases": [f.format(stem=stem) for f in spec["alias_forms"]],
                    "region": _region(rng), "people": _people(rng, 3, people), "crew": rng.randint(2, 12),
                    "code": f"{spec['code_prefix']}-{codes[i]}"})
    return out


def _margins(T: dict, t1: int, t2: int | None) -> tuple[int, int]:
    d, lv = T["drop"], T["level"]
    m1 = max(5 * d["step"], int(round(0.08 * t1 / d["step"])) * d["step"])
    ref = t2 if t2 is not None else (lv["t_lo"] + lv["t_hi"]) // 2
    m2 = max(3 * lv["step"], int(round(0.03 * ref / lv["step"])) * lv["step"])
    return m1, m2


def _readings(T: dict, drop: int, rng: random.Random) -> list[tuple[date, int]]:
    d = T["drop"]
    v1 = _draw(rng, max(d["lo"], d["lo"] + drop), min(d["hi"], d["hi"] + drop), d["step"])
    d1 = _rand_date(rng, date(YEAR, 2, 1), date(YEAR, 9, 30))
    return [(d1, v1), (d1 + timedelta(days=rng.randint(14, 70)), v1 - drop)]


def _drop(T: dict, t1: int, m1: int, sat: bool, rng: random.Random) -> int:
    d = T["drop"]
    if sat:
        return _draw(rng, t1 + m1, t1 + m1 + d["spread"], d["step"])
    while True:
        v = _draw(rng, -int(0.3 * t1), t1 - m1, d["step"])
        if v != 0:
            return v


def _level_range(T: dict) -> tuple[int, int]:
    lv = T["level"]
    return lv["base_lo"] + lv["off_lo"], lv["base_hi"] + lv["off_hi"]


def _split(T: dict, total: int, rng: random.Random, avoid_base: int | None = None) -> tuple[int, int]:
    lv = T["level"]
    for _ in range(100):
        off = _draw(rng, max(lv["off_lo"], total - lv["base_hi"]), min(lv["off_hi"], total - lv["base_lo"]), lv["step"])
        base = total - off
        if base != off and base != avoid_base:
            return base, off
    raise Regenerate("no split")


def _units(choices: list[str], rng: random.Random) -> str:
    return rng.choice(choices)


def sample_latent(T: dict, family: str, k: int, rng: random.Random) -> dict:
    if family == "join":
        types = ["A"] * (k // 2) + ["B"] * (k - k // 2)
    else:
        types = ["A"] * k
    ents = _entities(T, types, rng)
    d, lv = T["drop"], T["level"]
    t1 = _draw(rng, d["t_lo"], d["t_hi"], d["t_step"])
    t2 = _draw(rng, lv["t_lo"], lv["t_hi"], lv["t_step"]) if family == "agg" else None
    m1, m2 = _margins(T, t1, t2)
    lo_l, hi_l = _level_range(T)
    a = b = decoy = None
    if family == "agg":
        lo, hi = answer_range(k)
        n_ans = rng.randint(lo, hi)
        rest = k - n_ans
        q = max(1, rest // 4)
        pats = ["both"] * n_ans + ["c1"] * q + ["c2"] * q
        pats += [rng.choice(("c1", "c2", "none")) for _ in range(k - len(pats))]
        rng.shuffle(pats)
        for e, p in zip(ents, pats):
            e["pattern"] = p
            e["drop"] = _drop(T, t1, m1, p in ("both", "c1"), rng)
            total = _draw(rng, t2 + m2, hi_l, lv["step"]) if p in ("both", "c2") else _draw(rng, lo_l, t2 - m2, lv["step"])
            e["level"] = total
    else:
        A = [i for i, e in enumerate(ents) if e["type"] == "A"]
        B = [i for i, e in enumerate(ents) if e["type"] == "B"]
        a = rng.choice(A)
        decoy = rng.choice([i for i in A if i != a])
        gap = max(m2, 5 * lv["step"])
        levels: list[int] = []
        for _ in range(2000):
            if len(levels) == len(A):
                break
            v = _draw(rng, lo_l, hi_l, lv["step"])
            if all(abs(v - u) >= gap for u in levels):
                levels.append(v)
        if len(levels) < len(A):
            raise Regenerate("no distinct levels")
        for i, v in zip(A, levels):
            ents[i]["drop"] = _drop(T, t1, m1, i == a, rng)
            ents[i]["level"] = v
        perm = levels[:]
        rng.shuffle(perm)
        for i, v in zip(B, perm):
            ents[i]["drop"] = _drop(T, t1, m1, rng.random() < 0.3, rng)
            ents[i]["level"] = v
        b = next(i for i in B if ents[i]["level"] == ents[a]["level"])
    for e in ents:
        e["readings"] = _readings(T, e["drop"], rng)
        e["reading_units"] = [_units(d["units"], rng) for _ in range(2)]
        e["parts"] = _split(T, e["level"], rng)
        e["part_units"] = [_units(lv["base_units"], rng), _units(lv["off_units"], rng)]
    if family == "join":
        for i in B:
            twin = next(j for j in A if ents[j]["level"] == ents[i]["level"])
            ents[i]["parts"] = _split(T, ents[i]["level"], rng, avoid_base=ents[twin]["parts"][0])
    return {"ents": ents, "t1": t1, "t2": t2, "a": a, "b": b, "decoy": decoy, "m1": m1, "m2": m2}


def _plan_near(lat: dict, family: str, rng: random.Random) -> list[list[int]]:
    ents = lat["ents"]
    n = len(ents)
    want = [rng.randint(*NEAR) for _ in range(n)]
    targets: list[list[int]] = [[] for _ in range(n)]
    counts: Counter = Counter()
    decoy = lat["decoy"]
    if family == "join":
        hosts = rng.sample([i for i in range(n) if i != decoy], rng.randint(*DECOY_MENTIONS))
        for h in hosts:
            targets[h].append(decoy)
            counts[decoy] += 1
        cap = {i: (1 if i == lat["a"] else 2) for i, e in enumerate(ents) if e["type"] == "A" and i != decoy}
        cap[decoy] = 0
    else:
        cap = {}
    for i in range(n):
        while len(targets[i]) < want[i]:
            cands = [j for j in range(n) if j != i and j not in targets[i] and counts[j] < cap.get(j, 3)]
            if not cands:
                break
            j = rng.choice(cands)
            targets[i].append(j)
            counts[j] += 1
    return targets


def _ref(ent: dict, spec: dict, rng: random.Random, cap: bool = False) -> str:
    r = rng.random()
    s = rng.choice(spec["generic"]) if r < 0.4 else ent["name"] if r < 0.7 else rng.choice(ent["aliases"])
    return _cap(s) if cap else s


def _other(ent: dict, rng: random.Random) -> str:
    return ent["name"] if rng.random() < 0.6 else ent["aliases"][0]


def _content(T: dict, lat: dict, i: int, fmt: str, syn: dict, near: list[int], rng: random.Random) -> list[dict]:
    ents = lat["ents"]
    ent = ents[i]
    spec = T["types"][ent["type"]]
    short = fmt == "log"
    form = "log" if short else "prose"

    def slots(**kw) -> dict:
        s = {"ref": _ref(ent, spec, rng), "Ref": _ref(ent, spec, rng, cap=True), "kind": spec["kind"],
             "region": ent["region"], "lead": " ".join(ent["people"][0]), "code": ent["code"],
             "crew": fmt_count(ent["crew"], rng, short)}
        s.update(syn)
        s.update(kw)
        return s

    items = []
    d, lv = T["drop"], T["level"]
    for key, (dt, v), unit in zip(("drop1", "drop2"), ent["readings"], ent["reading_units"]):
        tpl = rng.choice(d[form])
        items.append({"key": key, "date": dt, "text": tpl.format(**slots(
            date=fmt_date(dt, rng.choice(DATE_STYLES)), qty=fmt_qty(v, unit, rng, short)))})
    for key, v, unit in zip(("base", "offset"), ent["parts"], ent["part_units"]):
        tpl = rng.choice(lv[f"{key}_{'log' if short else 'prose'}"])
        items.append({"key": key, "date": None, "text": tpl.format(**slots(qty=fmt_qty(v, unit, rng, short)))})
    for key in ("region", "lead", "crew", "code"):
        tpl = rng.choice(T["plain"][key][form])
        items.append({"key": key, "date": None, "text": tpl.format(**slots())})
    forbidden = {v for _, v in ent["readings"]} | set(ent["parts"]) | {lat["t1"], lat["t2"], ent["drop"], ent["level"]}
    n_d = rng.randint(*DISTRACTORS)
    oth = [x for x in T["distractors"] if x["other"]]
    fixed = [x for x in T["distractors"] if not x["other"] and not x["date"]]
    event = [x for x in T["distractors"] if not x["other"] and x["date"]]
    picks = [(rng.choice(oth), j) for j in near][:n_d]
    n_own = n_d - len(picks)
    n_fixed = rng.randint(min(n_own, max(1, len(fixed) - 1)), min(n_own, len(fixed)))
    picks += [(x, None) for x in rng.sample(fixed, n_fixed)]
    picks += [(rng.choice(event), None) for _ in range(n_own - n_fixed)]
    for tpl, j in picks:
        for _ in range(100):
            v = _draw(rng, tpl["lo"], tpl["hi"], tpl["step"])
            if v not in forbidden:
                break
        else:
            raise Regenerate("no distinct distractor value")
        forbidden.add(v)
        dt = _rand_date(rng) if tpl["date"] else None
        kw = {"qty": fmt_qty(v, rng.choice(tpl["units"]), rng, short)}
        if dt is not None:
            kw["date"] = fmt_date(dt, rng.choice(DATE_STYLES))
        if j is not None:
            kw["other"] = _other(ents[j], rng)
            kw["Other"] = _cap(kw["other"])
        items.append({"key": "distractor", "date": dt, "text": tpl[form].format(**slots(**kw)), "value": v,
                      "dim": tpl["dim"], "other": None if j is None else ents[j]["name"]})
    return items


def _filler(T: dict, rng: random.Random) -> str:
    f = T["filler"]
    return _cap(rng.choice(f["templates"]).format(**{k: rng.choice(v) for k, v in f["slots"].items()}))


def _noise(T: dict, rng: random.Random, dt: date | None = None) -> tuple[str, date | None]:
    f = T["noise"]
    tpls = f["templates"] if dt is None else [t for t in f["templates"] if "{date}" in t]
    tpl = rng.choice(tpls)
    slots = {k: rng.choice(v) for k, v in f["slots"].items()}
    if "{date}" in tpl:
        dt = dt or _rand_date(rng)
        slots["date"] = fmt_date(dt, rng.choice(DATE_STYLES))
    else:
        dt = None
    return tpl.format(**slots), dt


class _Doc:
    def __init__(self):
        self.lines: list[list[list]] = []
        self.n_fill = 0

    def fixed(self, text: str) -> int:
        self.lines.append([[-1, text]])
        return len(self.lines) - 1

    def slot(self) -> int:
        self.lines.append([])
        return len(self.lines) - 1

    def put(self, line: int, text: str, fill: bool = False) -> None:
        self.lines[line].append([self.n_fill if fill else -1, text])
        if fill:
            self.n_fill += 1

    def render(self, keep: int | None = None) -> str:
        keep = self.n_fill if keep is None else keep
        out = []
        for line in self.lines:
            kept = [t for o, t in line if o < keep]
            if kept or not line:
                out.append(" ".join(kept))
        return "\n".join(out)


def _apart(lines: list[int], rng: random.Random) -> tuple[int, int]:
    pairs = [(i, j) for i in range(len(lines)) for j in range(len(lines)) if abs(lines[i] - lines[j]) >= 2]
    if not pairs:
        raise Regenerate("no two slots apart")
    return rng.choice(pairs)


def _fill_to(doc: _Doc, slots: list[int], tok, target: int, est: int, rng: random.Random, make) -> None:
    while True:
        s = make()
        c = tok.count(s)
        if est + c > target:
            return
        doc.put(rng.choice(slots), s, fill=True)
        est += c


def _prose_doc(fmt: str, T: dict, ent: dict, items: list[dict], rng: random.Random, tok, target: int) -> _Doc:
    doc = _Doc()
    c = common()
    doc.fixed(T["titles"][fmt].format(name=ent["name"]))
    slots: list[int] = []
    if fmt == "report":
        slots = [doc.slot() for _ in range(rng.randint(8, 11))]
    else:
        people = ent["people"]
        dates = sorted(_rand_date(rng) for _ in range(3))
        subj = rng.choice(c["email"]["subjects"]).format(name=ent["name"])
        dom = T["email_domain"]
        for m in range(3):
            frm, to = people[m % 3], people[(m + 1) % 3]
            doc.fixed(f"From: {frm[0]} {frm[1]} <{frm[0].lower()}.{frm[1].lower()}@{dom}>")
            doc.fixed(f"To: {to[0]} {to[1]} <{to[0].lower()}.{to[1].lower()}@{dom}>")
            doc.fixed(f"Date: {fmt_date(dates[m], rng.choice(DATE_STYLES))}")
            doc.fixed(f"Subject: {'Re: ' if m else ''}{subj}")
            doc.fixed(rng.choice(c["email"]["greetings"]).format(first=to[0]))
            slots += [doc.slot() for _ in range(rng.randint(2, 4))]
            doc.fixed(rng.choice(c["email"]["signoffs"]).format(first=frm[0]))
            if m < 2:
                doc.fixed("")
    by = {it["key"]: it for it in items if it["key"] in ("drop1", "drop2", "base", "offset")}
    i1, i2 = _apart(slots, rng)
    j1, j2 = _apart(slots, rng)
    doc.put(slots[i1], by["drop1"]["text"])
    doc.put(slots[i2], by["drop2"]["text"])
    doc.put(slots[j1], by["base"]["text"])
    doc.put(slots[j2], by["offset"]["text"])
    for it in items:
        if it["key"] not in by:
            doc.put(rng.choice(slots), it["text"])
    for s in slots:
        doc.put(s, _filler(T, rng))
    est = sum(tok.count(t) for line in doc.lines for _, t in line)
    _fill_to(doc, slots, tok, int(target * OVERFILL), est, rng, lambda: _filler(T, rng))
    for s in slots:
        rng.shuffle(doc.lines[s])
    return doc


def _log_doc(T: dict, ent: dict, items: list[dict], rng: random.Random, tok, target: int) -> _Doc:
    spec = T["types"][ent["type"]]
    title = f"{spec['log_title']} | {ent['name']}"
    by = {it["key"]: it for it in items}
    d1, d2 = by["drop1"]["date"], by["drop2"]["date"]
    guard = _noise(T, rng, d1 + timedelta(days=rng.randint(1, (d2 - d1).days - 1)))
    dated = [(it["date"], -1, it["text"]) for it in items if it["date"] is not None] + [(guard[1], -1, guard[0])]
    undated = [(-1, it["text"], it["key"]) for it in items if it["date"] is None]
    est = tok.count(title) + sum(tok.count(t) for _, _, t in dated) + sum(tok.count(t) for _, t, _ in undated)
    n = 0
    while True:
        text, dt = _noise(T, rng)
        cost = tok.count(text)
        if est + cost > int(target * OVERFILL):
            break
        est += cost
        if dt is not None:
            dated.append((dt, n, text))
        else:
            undated.append((n, text, "noise"))
        n += 1
    dated.sort(key=lambda x: (x[0], rng.random()))
    seq: list[tuple[int, str, str]] = [(o, t, "dated") for _, o, t in dated]
    for o, t, key in undated:
        seq.insert(rng.randrange(len(seq) + 1), (o, t, key))
    keys = [k for _, _, k in seq]
    ib, io = keys.index("base"), keys.index("offset")
    lo, hi = min(ib, io), max(ib, io)
    if not any(seq[x][0] == -1 for x in range(lo + 1, hi)):
        g = _noise(T, rng)[0]
        seq.insert(lo + 1, (-1, g, "guard"))
    doc = _Doc()
    doc.fixed(title)
    for o, t, _ in seq:
        doc.lines.append([[o, t]])
    doc.n_fill = n
    return doc


def _fit(doc: _Doc, d: int, k: int, tok, target: int, budget: int) -> str:
    def size(keep: int) -> int:
        return tok.count(render_doc(d, k, doc.render(keep)))
    if size(0) > budget:
        raise Regenerate("content alone exceeds the document budget")
    lo, hi = 0, doc.n_fill
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if size(mid) <= target:
            lo = mid
        else:
            hi = mid - 1
    return doc.render(lo)


def part_lines(text: str, part) -> list[int]:
    """The lines of `text` that hold every [attribute, value] pair of one part of a needed sub-result."""
    return [i for i, line in enumerate(str(text).split("\n")) if all(N.value_in_text(v, a, line) for a, v in part)]


def span_problems(example: dict, needed: list[dict]) -> list[str]:
    out = []
    for n in needed:
        text = example["docs"][n["doc"] - 1]["text"]
        lines = []
        for p in n["parts"]:
            got = part_lines(text, p)
            if len(got) != 1:
                out.append(f"doc {n['doc']} {n['attribute']}: part {p} on lines {got}")
                break
            lines.append(got[0])
        else:
            if any(abs(x - y) < 2 for i, x in enumerate(lines) for y in lines[i + 1:]):
                out.append(f"doc {n['doc']} {n['attribute']}: parts on lines {lines}")
            every = [pair for p in n["parts"] for pair in p]
            if any(all(N.value_in_text(v, a, line) for a, v in every) for line in text.split("\n")):
                out.append(f"doc {n['doc']} {n['attribute']}: one line holds every value")
    return out


def other_mentions(example: dict, truth: dict) -> dict:
    """For each entity, how often it is named in documents other than its own."""
    own = {name: e["doc"] for name, e in truth["latent"]["entities"].items()}
    counts: Counter = Counter()
    for doc in example["docs"]:
        for name, c in N.mentions(example, doc["text"]).items():
            if own.get(name) != doc["doc"]:
                counts[name] += c
    return dict(counts)


def decoy_ok(example: dict, truth: dict) -> bool:
    if example.get("family") != "join":
        return True
    counts = other_mentions(example, truth)
    a_names = set(example["types"]["A"])
    a_counts = {n: counts.get(n, 0) for n in a_names}
    top = max(a_counts.values())
    leaders = [n for n, c in a_counts.items() if c == top]
    return top > 0 and len(leaders) == 1 and leaders[0] != truth["answer"][0]


def corpus_problems(example: dict, truth: dict, tok=None, solver: bool = True) -> list[str]:
    out = []
    k = example["k"]
    if [d["doc"] for d in example["docs"]] != list(range(1, k + 1)):
        out.append("documents are not numbered 1..k")
    fc = Counter(d["format"] for d in example["docs"])
    if max(fc.values()) - min(fc.get(f, 0) for f in FORMATS) > 1:
        out.append(f"formats unbalanced {dict(fc)}")
    out += span_problems(example, truth["needed"])
    lat = truth["latent"]
    if example["family"] == "agg":
        lo, hi = answer_range(k)
        if not lo <= len(truth["answer"]) <= hi:
            out.append(f"answer size {len(truth['answer'])}")
    else:
        sat = [n for n, e in lat["entities"].items() if e["type"] == "A" and e["drop"]["value"] > lat["t1"]]
        if sat != [truth["answer"][0]]:
            out.append(f"A entities meeting C1: {sat}")
        av = lat["entities"][truth["answer"][0]]["level"]["value"]
        match = [n for n, e in lat["entities"].items() if e["type"] == "B" and e["level"]["value"] == av]
        if match != [truth["answer"][1]]:
            out.append(f"B entities matching: {match}")
        if not decoy_ok(example, truth):
            out.append("decoy constraint")
    if solver and N.solve(example) != truth["answer"]:
        out.append(f"normaliser reads {N.solve(example)} not {truth['answer']}")
    planned = Counter(n for names in lat["near"].values() for n in names)
    if Counter(other_mentions(example, truth)) != planned:
        out.append("entity mentions differ from the plan")
    if tok is not None:
        sizes = [tok.count(render_doc(d["doc"], k, d["text"])) for d in example["docs"]]
        if max(sizes) > DOC_CAP:
            out.append(f"document over {DOC_CAP}: {max(sizes)}")
        if sum(sizes) > CORPUS_CAP:
            out.append(f"corpus over {CORPUS_CAP}: {sum(sizes)}")
        if sum(sizes) + tok.count(example["question"]) + ORACLE_OVERHEAD > ORACLE_CAP:
            out.append("oracle prompt over its cap")
    return out


def _units_of(fmt: str, text: str) -> list[tuple[int, int]]:
    lines = text.split("\n")
    if fmt == "report":
        return [(i, i + 1) for i in range(1, len(lines)) if lines[i].strip()]
    if fmt == "log":
        return [(i, min(i + 6, len(lines))) for i in range(1, len(lines), 6)]
    out = []
    for i, line in enumerate(lines):
        if line.startswith("Subject:"):
            body, end = i + 2, i + 2
            while end < len(lines) and lines[end] != "":
                end += 1
            if end - 1 > body:
                out.append((body, end - 1))
    return out


def unit_ok(example: dict, old: str, new) -> bool:
    if not isinstance(new, str) or not new.strip() or new.count("\n") != old.count("\n"):
        return False
    return N.value_multiset(new) == N.value_multiset(old) and N.mentions(example, new) == N.mentions(example, old)


def _call(paraphrase, text: str, seed: int) -> str:
    try:
        params = inspect.signature(paraphrase).parameters
    except (TypeError, ValueError):
        params = {}
    return paraphrase(text, seed=seed) if "seed" in params else paraphrase(text)


PARAPHRASE_WORKERS = 1


def _prefetch(example: dict, paraphrase, workers: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for doc in example["docs"]:
        lines = doc["text"].split("\n")
        for u, (s, e) in enumerate(_units_of(doc["format"], doc["text"])):
            seed = int(hashlib.sha256(f"{example['id']}|{doc['doc']}|{u}".encode()).hexdigest()[:8], 16)
            jobs.append(((doc["doc"], u), "\n".join(lines[s:e]), seed))

    def one(job):
        key, text, seed = job
        try:
            return key, _call(paraphrase, text, seed)
        except ParaphraseError as err:
            return key, err

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return dict(pool.map(one, jobs))


def paraphrase_corpus(example: dict, truth: dict, paraphrase, tok, budget: int, workers: int | None = None) -> dict:
    kept = rejected = errors = 0
    k = example["k"]
    workers = PARAPHRASE_WORKERS if workers is None else workers
    fetched = _prefetch(example, paraphrase, workers) if workers > 1 else None
    for doc in example["docs"]:
        orig = doc["text"]
        lines = orig.split("\n")
        n_doc = 0
        for u, (s, e) in enumerate(_units_of(doc["format"], orig)):
            old = "\n".join(lines[s:e])
            seed = int(hashlib.sha256(f"{example['id']}|{doc['doc']}|{u}".encode()).hexdigest()[:8], 16)
            try:
                if fetched is None:
                    new = _call(paraphrase, old, seed)
                else:
                    new = fetched[(doc["doc"], u)]
                    if isinstance(new, ParaphraseError):
                        raise new
            except ParaphraseError:
                errors += 1
                doc["text"] = orig
                return {"kept": kept, "rejected": rejected, "errors": errors}
            if not unit_ok(example, old, new):
                rejected += 1
                continue
            trial = lines[:s] + new.split("\n") + lines[e:]
            if tok.count(render_doc(doc["doc"], k, "\n".join(trial))) > budget:
                rejected += 1
                continue
            lines = trial
            n_doc += 1
        doc["text"] = "\n".join(lines)
        mine = [n for n in truth["needed"] if n["doc"] == doc["doc"]]
        if span_problems(example, mine):
            doc["text"] = orig
            rejected += n_doc
            n_doc = 0
        kept += n_doc
    return {"kept": kept, "rejected": rejected, "errors": errors}


def build_corpus(cid: str, family: str, domain: str, k: int, rng: random.Random, tok, paraphrase=None) -> tuple[dict, dict]:
    T = N.domain_templates(domain)
    lat = sample_latent(T, family, k, rng)
    ents = lat["ents"]
    near = _plan_near(lat, family, rng)
    order = list(range(k))
    rng.shuffle(order)
    doc_of = {i: d + 1 for d, i in enumerate(order)}
    fmts: dict[int, str] = {}
    for typ in sorted({e["type"] for e in ents}):
        idx = [i for i, e in enumerate(ents) if e["type"] == typ]
        f = [FORMATS[j % 3] for j in range(len(idx))]
        rng.shuffle(f)
        fmts.update(zip(idx, f))
    budget = doc_budget(k)
    target = min(TARGET_DOC_TOKENS, budget)
    docs = []
    for d in range(1, k + 1):
        i = order[d - 1]
        ent = ents[i]
        syn = {key: rng.choice(v) for key, v in T["syn"].items()}
        items = _content(T, lat, i, fmts[i], syn, near[i], rng)
        if fmts[i] == "log":
            doc = _log_doc(T, ent, items, rng, tok, target)
        else:
            doc = _prose_doc(fmts[i], T, ent, items, rng, tok, target)
        text = _fit(doc, d, k, tok, target, budget)
        docs.append({"doc": d, "title": text.split("\n", 1)[0], "format": fmts[i], "text": text})
    q_tpl = T["questions"][family]
    qrng = random.Random(f"{cid}|question")
    t1 = fmt_qty(lat["t1"], qrng.choice(T["drop"]["t_units"]), qrng, short=False)
    kw = {"t1": t1}
    if family == "agg":
        kw["t2"] = fmt_qty(lat["t2"], qrng.choice(T["level"]["t_units"]), qrng, short=False)
    example = {"id": cid, "family": family, "domain": domain, "k": k, "question": q_tpl.format(**kw), "docs": docs,
               "entities": {e["name"]: list(e["aliases"]) for e in ents}}
    if family == "join":
        example["types"] = {"A": [e["name"] for e in ents if e["type"] == "A"],
                            "B": [e["name"] for e in ents if e["type"] == "B"]}
    dim1, dim2 = T["drop"]["dim"], T["level"]["dim"]
    latent = {"family": family, "domain": domain, "k": k, "t1": lat["t1"], "t2": lat["t2"],
              "entities": {}, "near": {str(doc_of[i]): [ents[j]["name"] for j in near[i]] for i in range(k)}}
    needed = []
    for i, e in enumerate(ents):
        (dt1, v1), (dt2, v2) = e["readings"]
        latent["entities"][e["name"]] = {
            "doc": doc_of[i], "type": e["type"], "pattern": e.get("pattern"),
            "drop": {"readings": [[dt1.isoformat(), v1], [dt2.isoformat(), v2]], "value": e["drop"]},
            "level": {"base": e["parts"][0], "offset": e["parts"][1], "value": e["level"]},
            "region": e["region"], "lead": " ".join(e["people"][0]), "crew": e["crew"], "code": e["code"]}
        drop_entry = {"doc": doc_of[i], "attribute": T["drop"]["attribute"], "value": e["drop"],
                      "parts": [[["date", dt1.isoformat()], [dim1, v1]], [["date", dt2.isoformat()], [dim1, v2]]]}
        level_entry = {"doc": doc_of[i], "attribute": T["level"]["attribute"], "value": e["level"],
                       "parts": [[[dim2, e["parts"][0]]], [[dim2, e["parts"][1]]]]}
        if family == "agg":
            needed += [drop_entry, level_entry]
        elif e["type"] == "A":
            needed += [drop_entry] + ([level_entry] if i == lat["a"] else [])
        else:
            needed.append(level_entry)
    needed.sort(key=lambda n: (n["doc"], n["attribute"]))
    if family == "agg":
        answer = sorted(e["name"] for e in ents if e["pattern"] == "both")
        decoy = None
    else:
        answer = [ents[lat["a"]]["name"], ents[lat["b"]]["name"]]
        decoy = ents[lat["decoy"]]["name"]
        latent["a"], latent["b"] = answer
    truth = {"answer": answer, "latent": latent, "needed": needed, "decoy": decoy}
    problems = corpus_problems(example, truth, tok, solver=True)
    if problems:
        raise Regenerate("; ".join(problems[:3]))
    if paraphrase is not None:
        st = paraphrase_corpus(example, truth, paraphrase, tok, budget)
        if st["errors"]:
            raise ParaphraseError(f"{cid}: the paraphrase service failed; nothing is written")
        truth["latent"]["paraphrased_units"] = st["kept"]
        truth["latent"]["paraphrase_rejected"] = st["rejected"]
        truth["latent"]["paraphrase_errors"] = st["errors"]
        problems = corpus_problems(example, truth, tok, solver=False)
        if problems:
            raise Regenerate("after paraphrase: " + "; ".join(problems[:3]))
    return example, truth


def corpus_sha256(example: dict) -> str:
    return hashlib.sha256(json.dumps(example, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def generator_sha256() -> str:
    files = [HERE / "generate.py", HERE / "normalise.py"] + [p for p in TEMPLATES.glob("*.json") if p.is_file()]
    h = hashlib.sha256()
    for p in sorted(files, key=lambda p: p.relative_to(HERE).as_posix()):
        h.update(p.read_bytes())
    return h.hexdigest()


def id_of(prefix: str, family: str, num: int) -> str:
    return f"{prefix}_{num:04d}" if prefix.endswith("_train") else f"{prefix}_{family}_{num:04d}"


def _existing(out: Path) -> set:
    ids = {p.stem for p in (out / "corpora").glob("*.json")}
    t = out / "truth.json"
    if t.exists():
        ids |= set(json.loads(t.read_text(encoding="utf-8")))
    return ids


def _next_num(existing: set, prefix: str, family: str) -> int:
    head = id_of(prefix, family, 0)[:-4]
    nums = [int(i[len(head):]) for i in existing if i.startswith(head) and i[len(head):].isdigit()]
    return max(nums) + 1 if nums else 0


class _Lock:
    def __init__(self, path: Path, timeout: float = 600.0):
        self.path, self.timeout = path, timeout

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                os.mkdir(self.path)
                return self
            except FileExistsError:
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"lock {self.path} held for {self.timeout}s")
                time.sleep(0.2)

    def __exit__(self, *exc):
        os.rmdir(self.path)


def generate(out_dir, family: str, n: int, seed: int, tokenizer, paraphrase=None, k: int = K_DEFAULT,
             prefix: str = "ds") -> list[str]:
    if family not in FAMILIES:
        raise ValueError(family)
    if family == "join" and k % 2:
        raise ValueError("JOIN needs an even k")
    tok = tokenizer
    if isinstance(tok, str):
        from rci.agents.tokenizer import load_tokenizer
        tok = load_tokenizer(tok)
    out = Path(out_dir)
    (out / "corpora").mkdir(parents=True, exist_ok=True)
    start = _next_num(_existing(out), prefix, family)
    made: list[tuple[dict, dict]] = []
    for num in range(start, start + n):
        cid = id_of(prefix, family, num)
        domain = DOMAINS[num % len(DOMAINS)]
        last = None
        for attempt in range(MAX_ATTEMPTS):
            rng = random.Random(f"{seed}|{cid}|{attempt}")
            try:
                ex, tr = build_corpus(cid, family, domain, k, rng, tok, paraphrase)
            except Regenerate as e:
                last = e
                continue
            tr["latent"]["attempt"] = attempt
            made.append((ex, tr))
            break
        else:
            raise RuntimeError(f"{cid}: no admissible draw in {MAX_ATTEMPTS} attempts ({last})")
    with _Lock(out / "truth.json.lock"):
        clash = _existing(out) & {ex["id"] for ex, _ in made}
        if clash:
            raise FileExistsError(f"ids already generated: {sorted(clash)[:5]}")
        for ex, _ in made:
            path = out / "corpora" / f"{ex['id']}.json"
            with open(path, "x", encoding="utf-8") as f:
                json.dump(ex, f, ensure_ascii=False, sort_keys=True)
            os.chmod(path, 0o444)
        tpath = out / "truth.json"
        truth = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else {}
        for ex, tr in made:
            truth[ex["id"]] = tr
        tmp = tpath.with_suffix(f".json.tmp{os.getpid()}")
        tmp.write_text(json.dumps(truth, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
        os.chmod(tmp, 0o444)
        os.replace(tmp, tpath)
    return [ex["id"] for ex, _ in made]


def http_paraphraser(url: str, model: str, temperature: float = 0.7, max_tokens: int = 2048, timeout: float = 120.0,
                     tries: int = 4, backoff: float = 2.0, opener=None, sleep=time.sleep):
    import urllib.request
    opener = opener or urllib.request.urlopen

    prompt = ("Rewrite the text below in different words, keeping its meaning. Keep every number, unit, date, name and "
              "code exactly as written, keep the same number of lines, and do not add, drop or merge any fact. "
              "Reply with the rewritten text only.\n\n{text}")

    urls = [u.strip().rstrip("/") for u in url.split(",") if u.strip()]

    def fn(text: str, seed: int = 0) -> str:
        body = {"model": model, "messages": [{"role": "user", "content": prompt.format(text=text)}],
                "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(urls[seed % len(urls)] + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        last = None
        for attempt in range(max(1, tries)):
            if attempt:
                sleep(backoff * 2 ** (attempt - 1))
            try:
                with opener(req, timeout=timeout) as r:
                    out = json.loads(r.read().decode())
                return (out["choices"][0]["message"].get("content") or "").strip()
            except Exception as e:
                last = e
        raise ParaphraseError(f"paraphrase call failed after {max(1, tries)} attempts: {last!r}")
    return fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.docsweep.generate",
                                 description="Generate DocSweep corpora and questions.")
    ap.add_argument("--out", default=None, help="default: $RCI_DATA/docsweep")
    ap.add_argument("--family", choices=FAMILIES, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tokenizer", required=True, help="the writer model directory (or 'stub' for a dry run)")
    ap.add_argument("--paraphrase-url", default=None)
    ap.add_argument("--paraphrase-model", default=None)
    ap.add_argument("--prefix", default="ds")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--paraphrase-workers", type=int, default=1,
                    help="concurrent paraphrase calls per corpus (the same calls and seeds; only the waiting overlaps)")
    a = ap.parse_args(argv)
    a.out = a.out or str(paths.data_dir("docsweep"))
    global PARAPHRASE_WORKERS
    PARAPHRASE_WORKERS = max(1, a.paraphrase_workers)
    para = http_paraphraser(a.paraphrase_url, a.paraphrase_model) if a.paraphrase_url else None
    ids = generate(a.out, a.family, a.n, a.seed, a.tokenizer, paraphrase=para, k=a.k, prefix=a.prefix)
    summary = {"generated": len(ids), "first": ids[:1], "last": ids[-1:], "generator_sha256": generator_sha256()}
    if para is not None:
        truth = json.loads((Path(a.out) / "truth.json").read_text(encoding="utf-8"))
        lat = [truth[i]["latent"] for i in ids]
        summary["paraphrase"] = {key: sum(int(x.get(f, 0)) for x in lat) for key, f in
                                 (("kept", "paraphrased_units"), ("rejected", "paraphrase_rejected"),
                                  ("errors", "paraphrase_errors"))}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
