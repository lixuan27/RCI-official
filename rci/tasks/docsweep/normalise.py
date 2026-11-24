"""DocSweep normaliser: entity aliases, answers and the typed values written in the documents."""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent / "templates"
DOMAINS = ("field", "supply", "trial")
DIMS = ("volume", "length", "mass", "area")
BASE_UNIT = {"volume": "L", "length": "m", "mass": "kg", "area": "m2"}

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
_MON = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}
_MON.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)})
_MON["sept"] = 9
_MONTH_RE = "|".join(sorted(_MON, key=len, reverse=True))

_ISO = re.compile(r"(?<![\w-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\w-])")
_DMY = re.compile(rf"(?<!\w)(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?,?\s+(\d{{4}})(?!\w)", re.I)
_MDY = re.compile(rf"(?<!\w)({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})(?!\w)", re.I)
_CODE = re.compile(r"(?<![\w-])[A-Z]{2,4}-\d{2,6}(?![\w-])")

_UNITS = (
    (r"cubic\s+met(?:re|er)s?", "volume", 1000),
    (r"m³|m\^3|m3|cu\.?\s*m", "volume", 1000),
    (r"lit(?:re|er)s?", "volume", 1),
    (r"square\s+met(?:re|er)s?", "area", 1),
    (r"sq\.?\s*m", "area", 1),
    (r"m²|m\^2|m2", "area", 1),
    (r"hectares?", "area", 10000),
    (r"ha", "area", 10000),
    (r"kilomet(?:re|er)s?", "length", 1000),
    (r"km", "length", 1000),
    (r"met(?:re|er)s?", "length", 1),
    (r"kilogram(?:me)?s?", "mass", 1),
    (r"kgs?", "mass", 1),
    (r"tonnes?|metric\s+tons?", "mass", 1000),
    (r"l", "volume", 1),
    (r"m", "length", 1),
    (r"t", "mass", 1000),
)
_UNIT_RE = re.compile(r"\s*(?:" + "|".join(f"(?P<u{i}>{p})" for i, (p, _, _) in enumerate(_UNITS)) + r")(?![\w²³])",
                      re.I)

_SMALL = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALE = {"hundred": 100, "thousand": 1000}
_NUMWORD = "|".join(sorted(list(_SMALL) + list(_TENS) + list(_SCALE), key=len, reverse=True))
_WORDS_RE = re.compile(rf"(?<![\w-])(?:{_NUMWORD})(?:(?:\s+and\s+|[\s-]+)(?:{_NUMWORD}))*(?![\w-])", re.I)
_DIGITS_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])|(?<![\w.,])\d+(?:\.\d+)?(?![\d])")


def words_value(text: str) -> int | None:
    """The integer a run of English number words spells ("three hundred and forty" -> 340); None if it spells none."""
    total, cur, seen = 0, 0, False
    for w in re.split(r"[\s-]+", text.lower()):
        if not w or w == "and":
            continue
        if w in _SMALL:
            cur += _SMALL[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            cur = (cur or 1) * 100
        elif w == "thousand":
            total += (cur or 1) * 1000
            cur = 0
        else:
            return None
        seen = True
    return total + cur if seen else None


def number_words(n: int) -> str:
    """English words for 0 <= n < 1,000,000 (British "and" after the hundreds): 340 -> "three hundred and forty"."""
    if n < 0 or n >= 1_000_000:
        raise ValueError(n)
    small = list(_SMALL)
    tens = list(_TENS)

    def below_100(x: int) -> str:
        if x < 20:
            return small[x]
        t, u = divmod(x, 10)
        return tens[t - 2] + ("-" + small[u] if u else "")

    def below_1000(x: int) -> str:
        h, r = divmod(x, 100)
        if not h:
            return below_100(r)
        return small[h] + " hundred" + (" and " + below_100(r) if r else "")

    if n < 1000:
        return below_1000(n)
    th, r = divmod(n, 1000)
    out = below_1000(th) + " thousand"
    if r:
        out += (" and " if r < 100 else " ") + below_1000(r)
    return out


def _iso(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def _dates(text: str) -> list[tuple[int, int, str]]:
    out = []
    for m in _ISO.finditer(text):
        iso = _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if iso:
            out.append((m.start(), m.end(), iso))
    for m in _DMY.finditer(text):
        iso = _iso(int(m.group(3)), _MON[m.group(2).lower()], int(m.group(1)))
        if iso:
            out.append((m.start(), m.end(), iso))
    for m in _MDY.finditer(text):
        iso = _iso(int(m.group(3)), _MON[m.group(1).lower()], int(m.group(2)))
        if iso:
            out.append((m.start(), m.end(), iso))
    out.sort()
    keep, last = [], -1
    for s, e, v in out:
        if s >= last:
            keep.append((s, e, v))
            last = e
    return keep


def _mask(text: str, spans) -> str:
    chars = list(text)
    for s, e in spans:
        for i in range(s, e):
            chars[i] = " "
    return "".join(chars)


def _num(v: Decimal):
    v = v.normalize()
    return int(v) if v == v.to_integral_value() else float(v)


def typed_values(text: str) -> list[tuple[int, str, object]]:
    text = str(text or "")
    out: list[tuple[int, str, object]] = []
    dates = _dates(text)
    out += [(s, "date", v) for s, _, v in dates]
    codes = [(m.start(), m.end(), m.group(0)) for m in _CODE.finditer(text)]
    out += [(s, "code", v) for s, _, v in codes]
    masked = _mask(text, [(s, e) for s, e, _ in dates + codes])
    nums = []
    for m in _DIGITS_RE.finditer(masked):
        try:
            nums.append((m.start(), m.end(), Decimal(m.group(0).replace(",", ""))))
        except InvalidOperation:
            continue
    for m in _WORDS_RE.finditer(masked):
        w = words_value(m.group(0))
        if w is not None:
            nums.append((m.start(), m.end(), Decimal(w)))
    nums.sort()
    for s, e, v in nums:
        u = _UNIT_RE.match(masked, e)
        if u:
            i = next(int(k[1:]) for k, val in u.groupdict().items() if val is not None)
            _, dim, factor = _UNITS[i]
            out.append((s, dim, _num(v * factor)))
        else:
            out.append((s, "number", _num(v)))
    out.sort(key=lambda x: x[0])
    return out


@lru_cache(maxsize=None)
def domain_templates(domain: str) -> dict:
    """The template file of one domain (templates/<domain>.json)."""
    return json.loads((TEMPLATES / f"{domain}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def attribute_dims() -> dict:
    out = {d: d for d in DIMS}
    for dom in DOMAINS:
        t = domain_templates(dom)
        for key in ("drop", "level"):
            for name in ("attribute", "part", "base", "offset"):
                if name in t[key]:
                    out[t[key][name]] = t[key]["dim"]
    return out


def dim_of(attribute: str) -> str | None:
    """The dimension of an attribute name or dimension; None for "date", "code", "number" and plain attributes."""
    return attribute_dims().get(attribute)


def extract(text: str, attribute: str) -> list:
    vals = typed_values(text)
    if attribute == "all":
        return [(k, v) for _, k, v in vals]
    kind = attribute if attribute in ("date", "code", "number") else dim_of(attribute)
    if kind is None:
        raise ValueError(f"unknown attribute {attribute!r}")
    return [v for _, k, v in vals if k == kind]


def value_multiset(text: str) -> Counter:
    return Counter((k, json.dumps(v)) for k, v in extract(text, "all"))


def _close(a, b) -> bool:
    return abs(abs(float(a)) - abs(float(b))) <= 1e-6 * max(1.0, abs(float(b)))


def value_in_text(value, attribute: str, text: str) -> bool:
    if attribute == "date":
        return str(value) in extract(text, "date")
    if attribute == "code":
        return str(value) in extract(text, "code")
    dim = dim_of(attribute) if attribute != "number" else "number"
    if dim is not None:
        kind = dim
        return any(_close(x, value) for _, k, x in typed_values(text) if k == kind)
    return _ws(value).lower() in _ws(text).lower()


def entry_in_text(entry: dict, text: str) -> bool:
    """Whether `text` holds a needed sub-result of truth.json: its computed value, or every value of every part."""
    if value_in_text(entry["value"], entry["attribute"], text):
        return True
    parts = entry.get("parts") or []
    return bool(parts) and all(value_in_text(v, a, text) for part in parts for a, v in part)


def _ws(s) -> str:
    return re.sub(r"\s+", " ", str(s if s is not None else "")).strip()


def name_key(s) -> str:
    s = _ws(s).lower().strip(" .,:;!?\"'`()[]{}")
    return re.sub(r"^the\s+", "", s)


_ALIAS_CACHE: dict = {}


def _aliases(example: dict) -> tuple[dict, object]:
    ents = example.get("entities") or {}
    hit = _ALIAS_CACHE.get(id(ents))
    if hit is not None and hit[0] is ents:
        return hit[1], hit[2]
    table = {}
    for canon_name, aliases in ents.items():
        for a in [canon_name] + list(aliases or []):
            table[name_key(a)] = canon_name
    names = sorted(table, key=len, reverse=True)
    rx = None
    if names:
        alts = "|".join(r"\s+".join(re.escape(w) for w in n.split(" ")) for n in names)
        rx = re.compile(rf"(?<!\w)(?:the\s+)?({alts})(?!\w)", re.I)
    if len(_ALIAS_CACHE) >= 512:
        _ALIAS_CACHE.pop(next(iter(_ALIAS_CACHE)))
    _ALIAS_CACHE[id(ents)] = (ents, table, rx)
    return table, rx


def alias_table(example: dict) -> dict:
    """name_key(alias or canonical name) -> canonical name, over the corpus's entities."""
    return _aliases(example)[0]


def canon(example: dict, name) -> str:
    hit = alias_table(example).get(name_key(name))
    return hit if hit is not None else _ws(name).lower()


def names_of(answer) -> list[str]:
    if answer is None:
        return []
    if isinstance(answer, (list, tuple)):
        out = []
        for x in answer:
            if x is None:
                continue
            s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else str(x)
            if s.strip():
                out.append(s)
        return out
    if isinstance(answer, str):
        s = answer.strip()
        if s.startswith("["):
            try:
                v = json.loads(s)
            except ValueError:
                v = None
            if isinstance(v, list):
                return names_of(v)
        return [p for p in re.split(r"[,;|\n]", s) if p.strip()]
    return [str(answer)]


def pred_of(example: dict, answer) -> str | None:
    if answer is None:
        return None
    names = [canon(example, n) for n in names_of(answer)]
    if example.get("family") == "join":
        return "|".join(names)
    return json.dumps(sorted(set(names)), ensure_ascii=False)


def needed_values(example: dict, truth: dict) -> list[dict]:
    return [dict(n) for n in (truth or {}).get("needed") or []]


def mention_regex(example: dict):
    """One case-insensitive pattern matching every alias and canonical name of the corpus, longest first."""
    return _aliases(example)[1]


def mentions(example: dict, text: str) -> Counter:
    """How often each entity is named in `text` (by its canonical name or an alias)."""
    table, rx = _aliases(example)
    out: Counter = Counter()
    if rx is None:
        return out
    for m in rx.finditer(str(text or "")):
        out[table[name_key(m.group(1))]] += 1
    return out


def doc_entity(example: dict, text: str) -> str | None:
    """The entity a document is about: the one named in its first line."""
    first = str(text or "").split("\n", 1)[0]
    c = mentions(example, first)
    return c.most_common(1)[0][0] if c else None


def has_cue(line: str, cues) -> bool:
    """Whether `line` holds one of the cue phrases as whole words (case-insensitive)."""
    low = line.lower()
    return any(re.search(rf"(?<!\w){re.escape(c.lower())}(?!\w)", low) for c in cues)


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def sentences(text: str) -> list[str]:
    """The sentences of a text, line by line (a log line is one sentence)."""
    out = []
    for line in str(text or "").split("\n"):
        out += [s for s in _SENT.split(line) if s.strip()]
    return out


def read_doc(example: dict, text: str) -> dict:
    t = domain_templates(example["domain"])
    ent = doc_entity(example, text)
    d1, d2 = t["drop"]["dim"], t["level"]["dim"]
    bc, oc = t["level"]["base_cues"], t["level"]["offset_cues"]
    readings, base, off = [], [], []
    for s in sentences(str(text or "").split("\n", 1)[1] if "\n" in str(text or "") else ""):
        if any(e != ent for e in mentions(example, s)):
            continue
        vals = typed_values(s)
        dates = [v for _, k, v in vals if k == "date"]
        q1 = [v for _, k, v in vals if k == d1]
        q2 = [v for _, k, v in vals if k == d2]
        if has_cue(s, t["drop"]["cues"]) and len(dates) == 1 and len(q1) == 1:
            readings.append((dates[0], q1[0]))
        if len(q2) == 1 and has_cue(s, bc) and not has_cue(s, oc):
            base.append(q2[0])
        elif len(q2) == 1 and has_cue(s, oc) and not has_cue(s, bc):
            off.append(q2[0])
    drop = level = None
    if len(readings) == 2 and readings[0][0] != readings[1][0]:
        (_, early), (_, late) = sorted(readings)
        drop = early - late
    if len(base) == 1 and len(off) == 1:
        level = base[0] + off[0]
    return {"entity": ent, "readings": sorted(readings), "drop": drop, "base": base, "offset": off, "level": level}


def solve(example: dict, docs=None) -> list[str]:
    q = parse_question(example)
    want = list(docs) if docs is not None else [d["doc"] for d in example["docs"]]
    reads = [read_doc(example, example["docs"][d - 1]["text"]) for d in want]
    if q["family"] == "agg":
        return sorted(r["entity"] for r in reads if r["entity"] and r["drop"] is not None and r["level"] is not None
                      and q["t1"] is not None and q["t2"] is not None and r["drop"] > q["t1"] and r["level"] > q["t2"])
    types = example.get("types") or {}
    hits = [r for r in reads if r["entity"] in types.get("A", []) and r["drop"] is not None and q["t1"] is not None
            and r["drop"] > q["t1"]]
    if not hits:
        return []
    a = hits[0]
    match = [r for r in reads if r["entity"] in types.get("B", []) and r["level"] is not None and a["level"] is not None
             and _close(r["level"], a["level"])]
    return [a["entity"]] + ([match[0]["entity"]] if match else [])


def parse_question(example: dict) -> dict:
    t = domain_templates(example["domain"])
    q = example["question"]
    d1 = extract(q, t["drop"]["dim"])
    out = {"family": example["family"], "domain": example["domain"], "t1": d1[0] if d1 else None, "t2": None}
    if example["family"] == "agg":
        d2 = extract(q, t["level"]["dim"])
        out["t2"] = d2[0] if d2 else None
    return out
