"""Equal-budget ladder: vote curve V(n), pass@n curve T(n), budget n_B, levels and paired tests."""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

from rci.ladder.commit import _candidates, _vote_key, vote_stake


@dataclass
class Member:
    score: float
    key: str | None
    final: dict
    seat: int | None = None


def members_of(rec: dict) -> list[Member]:
    per = rec["score"]["per_agent"]
    return [Member(float(per[i]), _vote_key(rec, i), rec["finals"][i]["final"], i) for i in _candidates(rec)]


def _unstaked(final):
    return {**final, "confidence": None} if isinstance(final, dict) and final else final


def stakeless(rec: dict) -> dict:
    return {**rec, "finals": [{**f, "final": _unstaked(f.get("final"))} for f in rec.get("finals") or []]}


def commit(task: str, ms: list[Member], free: bool = False) -> float:
    rec = {"task": task, "score": {"per_agent": [m.score for m in ms], "preds": [m.key for m in ms]},
           "finals": [{"final": _unstaked(m.final) if free else m.final} for m in ms]}
    return float(vote_stake(rec))


def draw_curves(task: str, pool: list[Member], sizes: list[int], draws: int, rng: random.Random,
                free: bool = False) -> dict[int, tuple[float, float]]:
    """n -> (V, T) on one question, by drawing n members from its pool without replacement."""
    out = {}
    for n in sizes:
        k = min(n, len(pool))
        v = t = 0.0
        for _ in range(draws if k < len(pool) else 1):
            ms = rng.sample(pool, k)
            v += commit(task, ms, free)
            t += max(m.score for m in ms)
        d = draws if k < len(pool) else 1
        out[n] = (v / d, t / d)
    return out


ENUM_MAX = 5000
SEAT_DRAWS = 2000


def by_seat(pool: list[Member]) -> dict:
    """The pool split by seat, in seat order (members without a seat form one group)."""
    out: dict = {}
    for m in pool:
        out.setdefault(m.seat, []).append(m)
    return dict(sorted(out.items(), key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else 0)))


def t_drawn(groups: list[tuple[list[float], int]]) -> float:
    live = [(xs, min(k, len(xs))) for xs, k in groups if k > 0 and xs]
    if not live:
        return 0.0
    out = prev = 0.0
    for v in sorted({x for xs, _ in live for x in xs}):
        f = 1.0
        for xs, k in live:
            f *= math.comb(sum(1 for x in xs if x <= v), k) / math.comb(len(xs), k)
        out += v * (f - prev)
        prev = f
    return out


def seat_capacity(pool: list[Member]) -> tuple[int, int]:
    seats = by_seat(pool)
    least = min((len(g) for g in seats.values()), default=0)
    return len(seats) * least, least


def _seat_plans(seats: dict, n: int, balanced: bool = False) -> list[list[tuple[list[Member], int]]]:
    groups = list(seats.values())
    s = len(groups)
    q, r = divmod(n, s)
    if r and not (balanced or n < s):
        raise ValueError(f"{n} members do not split evenly over {s} seats")
    plans = [[(g, q + (i in pick)) for i, g in enumerate(groups) if q + (i in pick)] for pick in itertools.combinations(range(s), r)]
    if any(k > len(g) for plan in plans for g, k in plan):
        raise ValueError(f"a draw of {n} over {s} seats needs more members than the smallest seat holds")
    return plans


def _seat_point(task: str, seats: dict, n: int, draws: int, rng: random.Random, free: bool,
                balanced: bool = False) -> tuple[float, float]:
    plans = _seat_plans(seats, n, balanced)
    t = sum(t_drawn([([m.score for m in g], k) for g, k in plan]) for plan in plans) / len(plans)
    total = sum(math.prod(math.comb(len(g), k) for g, k in plan) for plan in plans)
    if total <= ENUM_MAX:
        v = 0.0
        for plan in plans:
            combos = list(itertools.product(*(itertools.combinations(g, k) for g, k in plan)))
            v += sum(commit(task, [m for part in c for m in part], free) for c in combos) / len(combos)
        return v / len(plans), t
    d = max(draws, SEAT_DRAWS)
    v = 0.0
    for _ in range(d):
        plan = plans[0] if len(plans) == 1 else rng.choice(plans)
        v += commit(task, [m for g, k in plan for m in rng.sample(g, k)], free)
    return v / d, t


def draw_curves_seats(task: str, pool: list[Member], sizes: list[int], draws: int, rng: random.Random,
                      free: bool = False, balanced: bool = False) -> dict[int, tuple[float, float]]:
    seats = by_seat(pool)
    s = len(seats)
    cap, _ = seat_capacity(pool)
    memo: dict[int, tuple[float, float]] = {}

    def point(n: int) -> tuple[float, float]:
        if n not in memo:
            memo[n] = _seat_point(task, seats, n, draws, rng, free, balanced)
        return memo[n]

    out = {}
    for n in sizes:
        if n > cap:
            continue
        if n < s or n % s == 0 or balanced:
            out[n] = point(n)
            continue
        lo, hi = s * (n // s), s * (n // s + 1)
        (v0, t0), (v1, t1) = point(lo), point(hi)
        w = (n - lo) / s
        out[n] = (v0 + w * (v1 - v0), t0 + w * (t1 - t0))
    return out


SIZES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)


def pass_curves(pool: dict, sizes=SIZES, seats: bool = True) -> dict:
    """question -> size -> exact pass@n of its independent members, drawn as evenly over the seats as n splits."""
    out = {}
    for q, mem in pool.items():
        cv = {}
        if seats:
            groups = by_seat(mem)
            cap, _ = seat_capacity(mem)
            for n in sizes:
                if n > cap:
                    continue
                plans = _seat_plans(groups, n, balanced=True)
                cv[n] = sum(t_drawn([([m.score for m in g], k) for g, k in plan]) for plan in plans) / len(plans)
        else:
            scores = [m.score for m in mem]
            cv = {n: t_drawn([(scores, n)]) for n in sizes if n <= len(scores)}
        if cv:
            out[q] = cv
    return out


def at(curve: dict[int, float], n: float) -> float:
    """The curve at a fractional size, linear between sizes, flat past the ends."""
    xs = sorted(curve)
    if n <= xs[0]:
        return curve[xs[0]]
    for a, b in zip(xs, xs[1:]):
        if n <= b:
            return curve[a] + (curve[b] - curve[a]) * (n - a) / (b - a)
    return curve[xs[-1]]


def effective(curve: dict[int, float], y: float) -> float | None:
    xs = sorted(curve)
    best = curve[xs[0]]
    if y <= best:
        return 0.0 if y < best else float(xs[0])
    for a, b in zip(xs, xs[1:]):
        top = max(best, curve[b])
        if y <= top:
            return a + (b - a) * (y - best) / (top - best) if top > best else float(b)
        best = top
    return None


def runs_worth(t: dict[int, float], y: float) -> float | None:
    return effective({0: 0.0, **t}, y)


def upper(mean: float, se: float) -> float:
    """The one-sided 95% upper bound of a paired difference."""
    return mean + 1.645 * se


def classify(mean: float, se: float, z: float = 1.96) -> str:
    """'below' when the difference is below zero by z SE, 'above' when above it by z SE, else 'undetermined'."""
    if mean + z * se < 0:
        return "below"
    if mean - z * se > 0:
        return "above"
    return "undetermined"


def p_one(mean: float, se: float) -> float:
    """The one-sided normal p-value of H0: difference <= 0 (a zero SE gives 0 for a positive mean, else 1)."""
    if se <= 0:
        return 0.0 if mean > 0 else 1.0
    return 0.5 * math.erfc(mean / se / math.sqrt(2))


def bh(ps: list) -> list:
    idx = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: ps[i])
    m = len(idx)
    out: list = [None] * len(ps)
    run = 1.0
    for rank in range(m, 0, -1):
        i = idx[rank - 1]
        run = min(run, ps[i] * m / rank)
        out[i] = run
    return out


def level(y: float, v1: float, v_b: float, t_b: float, margin: float = 0.0) -> int:
    """Highest level cleared by the point estimates minus one: -1 none, 0 V(1), 1 V(n_B), 2 T(n_B)."""
    if y <= v1 + margin:
        return -1
    if y <= v_b + margin:
        return 0
    if y <= t_b + margin:
        return 1
    return 2


UNITS = ("completion", "total")
BOOT_REPS = 2000
PASS_Z = 1.96
UB_Z = 1.645
TOST_Z = 1.645


def unit_cost(rec: dict, unit: str) -> float:
    """A record's cost in completion tokens or in total tokens."""
    c = rec.get("compute") or {}
    if unit == "completion":
        return float(c.get("completion_tokens", 0))
    if unit == "total":
        if c.get("total_tokens") is not None:
            return float(c["total_tokens"])
        return float(c.get("prompt_tokens", 0)) + float(c.get("completion_tokens", 0))
    raise ValueError(f"unknown unit {unit!r}")


def answering(rec: dict) -> int:
    return len(_candidates(rec))


def c1_per_question(pool: dict[str, list[dict]], unit: str) -> dict[str, float]:
    """Unit cost c1(q): the pool records' cost on q over their answering members."""
    out = {}
    for q, recs in pool.items():
        heads = sum(answering(r) for r in recs)
        if heads:
            out[q] = sum(unit_cost(r, unit) for r in recs) / heads
    return out


def pool_members(pool: dict[str, list[dict]]) -> dict[str, list[Member]]:
    out: dict[str, list[Member]] = {}
    for q, recs in pool.items():
        for r in recs:
            out.setdefault(q, []).extend(members_of(r))
    return {q: v for q, v in out.items() if v}


def correct(x) -> bool:
    """A member counts as right for T>=2 when it earned full credit (DocSweep scores are 0/1)."""
    return x is not None and float(x) >= 1.0


def t2_exact(scores: list[float], n: int) -> float:
    p = len(scores)
    k = min(n, p)
    if k < 2:
        return 0.0
    c = sum(1 for x in scores if correct(x))
    tot = math.comb(p, k)
    return 1.0 - math.comb(p - c, k) / tot - c * math.comb(p - c, k - 1) / tot


def t2_drawn(groups: list[tuple[list[float], int]]) -> float:
    dist = [1.0, 0.0, 0.0]
    for xs, k in groups:
        k = min(k, len(xs))
        if k <= 0 or not xs:
            continue
        p, c = len(xs), sum(1 for x in xs if correct(x))
        tot = math.comb(p, k)
        g0 = math.comb(p - c, k) / tot
        g1 = c * math.comb(p - c, k - 1) / tot
        g = [g0, g1, max(0.0, 1.0 - g0 - g1)]
        new = [0.0, 0.0, 0.0]
        for a in range(3):
            for b in range(3):
                new[min(2, a + b)] += dist[a] * g[b]
        dist = new
    return dist[2]


def v_point(task: str, members: list[Member], n: int, rng: random.Random, free: bool = False) -> float:
    k = min(n, len(members))
    if k <= 0:
        return 0.0
    if math.comb(len(members), k) <= ENUM_MAX:
        tot = cnt = 0
        for c in itertools.combinations(members, k):
            tot += commit(task, list(c), free)
            cnt += 1
        return tot / cnt
    return sum(commit(task, rng.sample(members, k), free) for _ in range(SEAT_DRAWS)) / SEAT_DRAWS


def exact_curves(task: str, members: list[Member], rng: random.Random, free: bool = False,
                 seats: bool = False) -> dict[int, tuple[float, float, float]]:
    if not members:
        return {}
    out = {}
    if not seats:
        scores = [m.score for m in members]
        for n in range(1, len(members) + 1):
            out[n] = (v_point(task, members, n, rng, free), t_drawn([(scores, n)]), t2_exact(scores, n))
        return out
    groups = by_seat(members)
    cap, _ = seat_capacity(members)
    for n in range(1, cap + 1):
        v, t = _seat_point(task, groups, n, SEAT_DRAWS, rng, free, balanced=True)
        plans = _seat_plans(groups, n, balanced=True)
        t2 = sum(t2_drawn([([m.score for m in g], k) for g, k in plan]) for plan in plans) / len(plans)
        out[n] = (v, t, t2)
    return out


def curve_value(curve: dict[int, tuple], x: float, which: int) -> float:
    return at({n: v[which] for n, v in curve.items()}, x)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs) -> float:
    """The sample standard deviation (n - 1); 0 for fewer than two values."""
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def boot_weights(n_q: int, reps: int = BOOT_REPS, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        w = [0] * n_q
        for _ in range(n_q):
            w[rng.randrange(n_q)] += 1
        out.append(w)
    return out


def paired(a: list[float], b: list[float], weights: list[list[int]] | None,
           b_reps: list[list[float]] | None = None) -> dict:
    d = [x - y for x, y in zip(a, b)]
    q = len(d)
    if not q:
        return {"mean": float("nan"), "se": float("nan"), "se_analytic": float("nan"), "se_boot": float("nan"), "q": 0}
    m = sum(d) / q
    se_a = sd(d) / math.sqrt(q)
    se_b = 0.0
    if weights:
        stats = []
        for r, w in enumerate(weights):
            bb = b_reps[r] if b_reps is not None else b
            tot = sum(w)
            stats.append(sum(wi * (x - y) for wi, x, y in zip(w, a, bb)) / tot)
        se_b = sd(stats)
    return {"mean": m, "se": max(se_a, se_b), "se_analytic": se_a, "se_boot": se_b, "q": q}


def passes(c: dict, z: float = PASS_Z) -> bool:
    return c["mean"] - z * c["se"] > 0


def below(c: dict, z: float = PASS_Z) -> bool:
    """mean + 1.96 SE < 0."""
    return c["mean"] + z * c["se"] < 0


def ub(c: dict) -> float:
    return c["mean"] + UB_Z * c["se"]


def tost(mean_: float, se: float, delta: float, z: float = PASS_Z) -> str:
    if mean_ - z * se > 0:
        return "above"
    if mean_ + z * se < 0:
        return "below"
    if mean_ - TOST_Z * se > -delta and mean_ + TOST_Z * se < delta:
        return "equivalent"
    return "undetermined"


def percentile_high(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if not ys:
        return float("nan")
    k = max(1, math.ceil(p / 100.0 * len(ys) - 1e-12))
    return ys[min(k, len(ys)) - 1]


def ladder(task: str, group: dict[str, list[tuple[float, float, float]]], curves: dict[str, dict[int, tuple]],
           weights: list[list[int]] | None = None) -> dict:
    qs = sorted(q for q in group if q in curves and group[q])
    if not qs:
        return {"questions": 0}
    rows = {k: [] for k in ("y", "v1", "vb", "tb", "t2b", "tmax", "pass")}
    nbs = []
    for q in qs:
        cv = curves[q]
        top = max(cv)
        recs = group[q]
        rows["y"].append(mean(y for y, _, _ in recs))
        rows["pass"].append(mean(p for _, p, _ in recs))
        rows["v1"].append(cv[min(cv)][0])
        rows["vb"].append(mean(curve_value(cv, nb, 0) for _, _, nb in recs))
        rows["tb"].append(mean(curve_value(cv, nb, 1) for _, _, nb in recs))
        rows["t2b"].append(mean(curve_value(cv, nb, 2) for _, _, nb in recs))
        rows["tmax"].append(cv[top][1])
        nbs += [nb for _, _, nb in recs]
    w = weights if weights is not None else boot_weights(len(qs))
    if len(w) and len(w[0]) != len(qs):
        w = boot_weights(len(qs), len(w))
    y = rows["y"]
    e = {"e_one": paired(y, rows["v1"], w), "e_vote": paired(y, rows["vb"], w), "e_truth": paired(y, rows["tb"], w),
         "e_truth2": paired(y, rows["t2b"], w), "e_max": paired(y, rows["tmax"], w)}
    nb = mean(nbs)
    pool_n = min(max(curves[q]) for q in qs)
    capped = nb > pool_n

    def climb(third: str) -> int:
        got = 0
        for k in (("e_one", "e_vote") if capped else ("e_one", "e_vote", third)):
            if not passes(e[k]):
                break
            got += 1
        return got

    lv, lv2 = climb("e_truth"), climb("e_truth2")
    return {"questions": len(qs), "qs": qs, "answer": mean(y), "pass": mean(rows["pass"]), "n_budget": nb,
            "nb_records": nbs, "pool_n": pool_n, "capped": capped, "v1": mean(rows["v1"]), "v_b": mean(rows["vb"]),
            "t_b": mean(rows["tb"]), "t2_b": mean(rows["t2b"]), "t_max": mean(rows["tmax"]), **e, "level": lv,
            "level_t2": lv2, "level3": None if capped else lv >= 3, "level3_t2": None if capped else lv2 >= 3,
            "per_q": {k: dict(zip(qs, v)) for k, v in rows.items()}}


def coverage_ok(x_top: float, group_nb: list[float]) -> bool:
    return bool(group_nb) and not x_top < percentile_high(group_nb, 90)


def z_of(c: dict) -> float | None:
    se = c.get("se")
    if se is None or not se or c.get("mean") is None:
        return None
    return c["mean"] / se
