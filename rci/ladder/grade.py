"""Grade every cell of a spec on the equal-budget ladder against pools of independent agents."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from rci.ladder import curves as lc
from rci.ladder.commit import _candidates
from rci.ladder.curves import (SIZES, at, bh, classify, draw_curves, draw_curves_seats, members_of, p_one,
                               runs_worth, seat_capacity, upper)
from rci.ladder.records import answer_of, cost_of, records, review_cost

MIN_QUESTIONS = 40
BOOTS = 400
COSTS = ("completion", "total", "agents")
PQ_UNITS = ("completion", "total")
QS = ("q_vote_task", "q_vote_run", "q_truth_task", "q_truth_run")
FAMILIES = ("all", "census")


def curves(task: dict, pool: dict, rng: random.Random, draws: int, seats: bool = False, free: bool = False,
           balanced: bool = False) -> dict:
    """question -> size -> (V, T) for one pool of independent members."""
    out = {}
    grid = tuple(task.get("sizes") or SIZES)
    for q, ms_ in pool.items():
        sizes = [n for n in grid if n <= len(ms_)]
        if seats:
            out[q] = draw_curves_seats(task["task"], ms_, sizes, draws, rng, free, balanced)
        else:
            out[q] = draw_curves(task["task"], ms_, sizes, draws, rng, free)
    return out


def ms(xs: list) -> tuple:
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1) / len(xs))


def _climb(contrasts) -> int:
    got = 0
    for mean_, se in contrasts:
        if mean_ - 1.96 * se <= 0:
            break
        got += 1
    return got


def _eta_ci(cs: dict, qs: list, ys: dict, common: list, nb: float) -> list:
    gains = []
    boot = random.Random(0)
    for _ in range(BOOTS):
        pick_ = [boot.choice(qs) for _ in qs]
        tb = {n: sum(cs[q][n][1] for q in pick_) / len(pick_) for n in common}
        e = runs_worth(tb, sum(ys[q] for q in pick_) / len(pick_))
        gains.append(math.inf if e is None else e / nb)
    gains.sort()
    return [gains[int(0.025 * BOOTS)], gains[int(0.975 * BOOTS)]]


def _row(label, qs, y, passes, mems, common, v, t, nb, ev, et, e1, nt, eta_ci, lv, efull, capped, over, pools, need,
         seat_min) -> dict:
    return {"cell": label, "questions": len(qs), "answer": y, "pass": sum(passes[q] for q in qs) / len(qs),
            "member": sum(mems[q] for q in qs) / len(qs),
            "curve": {"n": common, "v": [v[n] for n in common], "t": [t[n] for n in common]},
            "n_budget": nb, "v1": v[common[0]], "v_b": at(v, nb), "t_b": at(t, nb), "t_full": t[common[-1]],
            "pool_n": common[-1], "e_vote": ev, "e_truth": et, "e_one": e1, "n_t": nt,
            "eta": None if nt is None else nt / nb, "eta_ci": eta_ci, "level": lv, "margin_full": efull,
            "ub_one": upper(*e1), "ub_vote": upper(*ev), "ub_truth": None if capped else upper(*et),
            "ub_full": None if capped else upper(*efull), "class_vote": classify(*ev),
            "class_truth": "capped" if capped else classify(*et), "p_vote": p_one(*ev),
            "p_truth": None if capped else p_one(*et), "pool_size": min(pools.values()),
            "short_q": sum(1 for q in qs if pools[q] < need), "over_pool": over, "capped": capped,
            "min_seat": min(seat_min[q] for q in qs) if seat_min else None}


def measure(task: dict, label: str, recs: dict, cs: dict, c1: float, cost: str, review: Path | None,
            least: int = MIN_QUESTIONS, scale: float = 1.0, *, sizes: dict | None = None, strict: bool = False,
            free: bool = False, seat_min: dict | None = None) -> dict | None:
    """One cell against its pool curves at the cell's mean budget n_B."""
    ys, passes, mems, spend = {}, {}, {}, []
    for q, rr in recs.items():
        if q not in cs:
            continue
        got = [(answer_of(r, False, review, free), r) for r in rr]
        got = [(y, r) for y, r in got if y is not None]
        if not got:
            continue
        ys[q] = sum(y for y, _ in got) / len(got)
        passes[q] = sum(float(max(x for x in r["score"]["per_agent"] if x is not None)) for _, r in got) / len(got)
        pa = [x for _, r in got for x in r["score"]["per_agent"] if x is not None]
        mems[q] = sum(float(x) for x in pa) / len(pa)
        spend += [cost_of(r, cost) + review_cost(r, review, cost, scale) for _, r in got]
    if len(ys) < least:
        return None
    qs = sorted(ys)
    nb = sum(spend) / len(spend) / c1
    common = sorted(set.intersection(*(set(cs[q]) for q in qs)))
    v = {n: sum(cs[q][n][0] for q in qs) / len(qs) for n in common}
    t = {n: sum(cs[q][n][1] for q in qs) / len(qs) for n in common}
    y = sum(ys[q] for q in qs) / len(qs)

    def d(ref, n):
        return ms([ys[q] - at({k: cs[q][k][ref] for k in common}, n) for q in qs])

    ev, et, e1 = d(0, nb), d(1, nb), d(0, 1)
    efull = d(1, common[-1])
    over = nb > common[-1]
    capped = bool(strict and over)
    lv = _climb((e1, ev) if capped else (e1, ev, et))
    nt = runs_worth(t, y)
    pools = {q: (sizes[q] if sizes and q in sizes else max(cs[q])) for q in qs}
    return _row(label, qs, y, passes, mems, common, v, t, nb, ev, et, e1, nt, _eta_ci(cs, qs, ys, common, nb), lv,
                efull, capped, over, pools, math.ceil(nb - 1e-9), seat_min)


def measure_pq(task: dict, label: str, recs: dict, cs: dict, c1q: dict, cost: str, least: int, free: bool,
               seat_min: dict | None = None) -> dict | None:
    """One cell of a per-question task: n_B per record against c1(q), curves read at each record's n_B."""
    group, mems = {}, {}
    for q in sorted(recs):
        if q not in cs or q not in c1q or not cs[q]:
            continue
        got, pa = [], []
        for r in recs[q]:
            nb = lc.unit_cost(r, cost) / c1q[q]
            per = [x for x in r["score"]["per_agent"] if x is not None]
            got.append((answer_of(r, False, None, free), float(max(per)) if per else 0.0, nb))
            pa += [float(x) for x in per]
        group[q] = got
        mems[q] = sum(pa) / len(pa) if pa else 0.0
    if len(group) < least:
        return None
    lad = lc.ladder(task["task"], group, cs)
    qs, nb, capped, lv = lad["qs"], lad["n_budget"], lad["capped"], lad["level"]
    common = list(range(1, lad["pool_n"] + 1))
    v = {n: sum(cs[q][n][0] for q in qs) / len(qs) for n in common}
    t = {n: sum(cs[q][n][1] for q in qs) / len(qs) for n in common}
    ys = lad["per_q"]["y"]
    y = lad["answer"]
    nt = runs_worth(t, y)
    tup = {k: (lad[k]["mean"], lad[k]["se"]) for k in ("e_one", "e_vote", "e_truth", "e_max")}
    pools = {q: max(cs[q]) for q in qs}
    r = _row(label, qs, y, lad["per_q"]["pass"], mems, common, v, t, nb, tup["e_vote"], tup["e_truth"], tup["e_one"],
             nt, _eta_ci(cs, qs, ys, common, nb), lv, tup["e_max"], capped, nb > lad["pool_n"], pools,
             math.ceil(nb - 1e-9), seat_min)
    r.update(v1=lad["v1"], v_b=lad["v_b"], t_b=lad["t_b"], t_full=lad["t_max"], per_question=True)
    return r


def fq(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def fs(x) -> str:
    return "-" if x is None else f"{x:+.3f}"


def kind_of(r: dict) -> str:
    """group, null (a pool cell graded without its own runs), review, or review x<scale>."""
    if r.get("null"):
        return "null"
    if r.get("review"):
        scale = r.get("review_scale", 1.0)
        return "review" if scale == 1.0 else f"review x{scale:g}"
    return "group"


def in_family(r: dict, kind: str, family: str) -> bool:
    if r.get("stops") or r.get(f"p_{kind}") is None or (kind == "truth" and r.get("capped")):
        return False
    if r.get("bh") is False:
        return False
    return family == "all" or not (r.get("null") or r.get("review"))


def add_q(rows: list, scope: str, family: str = "all") -> dict:
    """Benjamini-Hochberg q-values of levels 2 and 3 over the rows, one family per level."""
    m = {}
    for kind in ("vote", "truth"):
        live = [r for r in rows if in_family(r, kind, family)]
        for r in rows:
            r[f"q_{kind}_{scope}"] = None
        for r, q in zip(live, bh([r[f"p_{kind}"] for r in live])):
            r[f"q_{kind}_{scope}"] = q
        m[f"{kind}_{scope}"] = len(live)
    return m


def add_pass(rows: list, alpha: float | None) -> None:
    if alpha is None:
        return
    for r in rows:
        r["bh_pass"] = {k[2:]: (r.get(k) is not None and r[k] <= alpha) for k in QS}


def row_md(r: dict) -> str:
    ne = f">{r['pool_n']}" if r["n_t"] is None else f"{r['n_t']:.1f}"
    eta = ">pool" if r["eta"] is None else f"{r['eta']:.2f}"
    low, high = r["eta_ci"]
    eta += f" [{low:.2f}, {'inf' if high == math.inf else f'{high:.2f}'}]"
    if r.get("stops"):
        ne, eta = "-", "stops early"
    lv = f"{r['level']} capped" if r.get("capped") and r["level"] is not None else f"{r['level']}"
    return (f"| {r['cell']} | {r['questions']} | {r['n_budget']:.1f} | {r['answer']:.3f} | {r['v_b']:.3f} | "
            f"{r['t_b']:.3f} | {r['e_vote'][0]:+.3f} +- {r['e_vote'][1]:.3f} | {r['e_truth'][0]:+.3f} +- "
            f"{r['e_truth'][1]:.3f} | {ne} | {eta} | {lv} |")


def detail_row(r: dict, alpha: float | None, seats: bool = False) -> str:
    stop = bool(r.get("stops"))
    marks = [kind_of(r)] + (["stops"] if stop else []) + (["capped"] if r.get("capped") else [])
    cells = [r["cell"], r.get("root", ""), ", ".join(marks), f"{r['n_budget']:.1f}", f"{r['pool_n']}",
             f"{r['pool_size']}", *([f"{r.get('min_seat')}"] if seats else []), f"{r['short_q']}",
             "yes" if r.get("over_pool") else "no", "-" if stop else fs(r["ub_vote"]),
             "-" if stop else r["class_vote"], "-" if stop else fs(r["ub_truth"]), "-" if stop else r["class_truth"],
             *(fq(r.get(k)) for k in QS)]
    if alpha is not None:
        bp = r.get("bh_pass") or {}
        for kind in ("vote", "truth"):
            got = [s for s in ("task", "run") if bp.get(f"{kind}_{s}")]
            cells.append("+".join(got) or "-")
    return "| " + " | ".join(cells) + " |"


def full_pool_section(rows: list) -> list:
    """Y - T(n_max) for the rows at level 3: descriptive, never a level."""
    top = [r for r in rows if r.get("level") == 3 and not r.get("stops")]
    if not top:
        return []
    head = ["cell", "root", "level", "pool_n", "T(n_max)", "Y - T(n_max)", "ub (1.645 SE)", "class (1.96 SE)"]
    out = ["", "### margin over the full pool (descriptive, not a level)", "",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in top:
        m, se = r["margin_full"]
        out.append(f"| {r['cell']} | {r.get('root', '')} | 3 | {r['pool_n']} | {r['t_full']:.3f} | {m:+.3f} +- {se:.3f} | "
                   f"{fs(r.get('ub_full'))} | {classify(m, se)} |")
    return out


def render(name: str, res: dict, cost: str, alpha: float | None) -> str:
    unit = (f"{res['c1']:.2f} {cost} per answering member" if cost == "agents"
            else f"{res['c1'] / 1e3:.1f}k {cost} tokens")
    text = [f"## {name}", "", f"independent run: {unit}, mean pool size {res['pool_mean']:.0f}", "",
            "| cell | q | n_B | Y | V(n_B) | T(n_B) | Y - V | Y - T | n_T | eta | level |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    text += [row_md(r) for r in res["rows"]]
    seats = bool(res["options"].get("stratify_seats"))
    head = (["cell", "root", "kind", "n_B", "pool_n", "pool"] + (["min seat"] if seats else [])
            + ["short", "n_B > pool_n", "Y - V ub (1.645 SE)", "V class (1.96 SE)", "Y - T ub (1.645 SE)",
               "T class (1.96 SE)", "q_V task", "q_V run", "q_T task", "q_T run"]
            + (["BH V", "BH T"] if alpha is not None else []))
    text += ["", "### details", "", *(f"- {x}" for x in res.get("notes", [])), *([""] if res.get("notes") else []),
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    text += [detail_row(r, alpha, seats) for r in res["rows"]]
    text += full_pool_section(res["rows"])
    return "\n".join(text) + "\n"


def write(out: Path, name: str, tag: str, res: dict, cost: str, alpha: float | None) -> str:
    text = render(name, res, cost, alpha)
    (out / f"{name}_{tag}.md").write_text(text)
    body = {"c1": res["c1"], "rows": res["rows"], "options": res["options"], "answer_rate": res["answer_rate"]}
    (out / f"{name}_{tag}.json").write_text(json.dumps(body, indent=1))
    return text


def without(pool: dict, src: dict, key) -> dict:
    """The pool without the runs of one of its own cells (the null test)."""
    rest = {q: [m for m, s in zip(pool[q], src[q]) if s != key] for q in pool}
    return {q: v for q, v in rest.items() if v}


def pool_sizes(pool: dict, seats: bool) -> tuple:
    if not seats:
        return {q: len(v) for q, v in pool.items()}, None
    caps = {q: seat_capacity(v) for q, v in pool.items()}
    return {q: c[0] for q, c in caps.items()}, {q: c[1] for q, c in caps.items()}


def _cell_key(cell: dict) -> tuple:
    return cell["root"], cell["protocol"], cell["n"]


def _check_widths(name: str, recs_by_cell) -> None:
    widths = {len(r.get("finals") or []) for recs in recs_by_cell for rr in recs.values() for r in rr}
    if len(widths) > 1:
        raise SystemExit(f"{name}: --stratify-seats needs pool records of one group size, got {sorted(widths)}")


def per_question_task(name: str, task: dict, a, rng: random.Random, free: bool, seats: bool) -> dict | None:
    if a.cost not in PQ_UNITS:
        print(f"[grade] {name}: per-question tasks are priced in completion or total tokens; skipped", flush=True)
        return None
    pool_recs: dict = {}
    keys: dict = {}
    cells = {}
    for root, protocol, n in task["pool"]:
        cells[(root, protocol, n)] = records(root, protocol, n)
        for q, rr in cells[(root, protocol, n)].items():
            pool_recs.setdefault(q, []).extend(rr)
    if seats:
        _check_widths(name, [pool_recs])
    c1q = lc.c1_per_question(pool_recs, a.cost)
    pool: dict = {}
    for key, recs in cells.items():
        for q, rr in recs.items():
            for r in rr:
                got = members_of(r)
                pool.setdefault(q, []).extend(got)
                keys.setdefault(q, []).extend([key] * len(got))
    if task.get("min_pool"):
        pool = {q: v for q, v in pool.items() if len(v) >= task["min_pool"]}
    count = sum(len(_candidates(r)) for rr in pool_recs.values() for r in rr)
    heads = sum(cost_of(r, "agents") for rr in pool_recs.values() for r in rr)

    def build(pl):
        return {q: lc.exact_curves(task["task"], pl[q], rng, free, seats) for q in sorted(pl)}

    def mins(pl):
        return {q: seat_capacity(pl[q])[1] for q in pl} if seats else None

    base = build(pool)
    rows = []
    for cell in task["cells"]:
        label, key = cell["label"], _cell_key(cell)
        recs = records(*key)
        cs, seat_min = base, mins(pool)
        if key in cells:
            rest = without(pool, keys, key)
            cs, seat_min = build(rest), mins(rest)
        r = measure_pq(task, label, recs, cs, c1q, a.cost, task.get("least", MIN_QUESTIONS), free, seat_min)
        if r is None:
            print(f"[grade] {name} {label}: too few questions (found {len(recs)}, need "
                  f"{task.get('least', MIN_QUESTIONS)})", flush=True)
            continue
        r.update(root=key[0], protocol=key[1], n=key[2], null=key in cells, review=None, review_scale=1.0)
        if cell.get("bh") is False:
            r["bh"] = False
        if cell.get("stops"):
            r.update(level=None, eta=None, n_t=None, stops=True)
        rows.append(r)
        print(f"[grade] {name} {label} n_B {r['n_budget']:.1f} Y {r['answer']:.3f} V {r['v_b']:.3f} "
              f"T {r['t_b']:.3f}", flush=True)
    c1_mean = sum(c1q[q] for q in pool if q in c1q) / max(sum(1 for q in pool if q in c1q), 1)
    return {"rows": rows, "pool": pool, "count": count, "heads": heads, "c1": c1_mean, "c1_q": c1q}


def standard_task(name: str, task: dict, a, rng: random.Random, free: bool, seats: bool, balanced: bool) -> dict:
    cells = {}
    for root, protocol, n in task["pool"]:
        cells[(root, protocol, n)] = records(root, protocol, n)
    if seats:
        _check_widths(name, cells.values())
    pool: dict = {}
    src: dict = {}
    tokens, count, heads = 0.0, 0, 0.0
    for key, recs in cells.items():
        for q, rr in recs.items():
            for r in rr:
                got = members_of(r)
                pool.setdefault(q, []).extend(got)
                src.setdefault(q, []).extend([key] * len(got))
                tokens += cost_of(r, a.cost)
                count += len(_candidates(r))
                heads += cost_of(r, "agents")
    c1 = tokens / max(count, 1)
    if task.get("min_pool"):
        pool = {q: v for q, v in pool.items() if len(v) >= task["min_pool"]}
    base = curves(task, pool, rng, a.draws, seats, free, balanced)
    base_sizes, base_seat = pool_sizes(pool, seats)
    rows = []
    for cell in task["cells"]:
        label, key = cell["label"], _cell_key(cell)
        recs = records(*key)
        review = Path(cell["review"]) if cell.get("review") else None
        cs, sizes, seat_min = base, base_sizes, base_seat
        if key in cells:
            rest = without(pool, src, key)
            cs = curves(task, rest, rng, a.draws, seats, free, balanced)
            sizes, seat_min = pool_sizes(rest, seats)
        r = measure(task, label, recs, cs, c1, a.cost, review, task.get("least", MIN_QUESTIONS),
                    cell.get("review_scale", 1.0), sizes=sizes, strict=a.strict_pool, free=free, seat_min=seat_min)
        if r is None:
            print(f"[grade] {name} {label}: too few questions (found {len(recs)}, need "
                  f"{task.get('least', MIN_QUESTIONS)})", flush=True)
            continue
        r.update(root=key[0], protocol=key[1], n=key[2], null=key in cells, review=cell.get("review"),
                 review_scale=cell.get("review_scale", 1.0))
        if cell.get("bh") is False:
            r["bh"] = False
        if cell.get("stops"):
            r.update(level=None, eta=None, n_t=None, stops=True)
        rows.append(r)
        print(f"[grade] {name} {label} n_B {r['n_budget']:.1f} Y {r['answer']:.3f} V {r['v_b']:.3f} "
              f"T {r['t_b']:.3f}", flush=True)
    return {"rows": rows, "pool": pool, "count": count, "heads": heads, "c1": c1}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grade every cell of a spec on the equal-budget ladder: the cell's accuracy "
                                             "Y against the vote V and pass@n T of independent agents at the same budget.")
    ap.add_argument("spec")
    ap.add_argument("--out", default="results/ladder")
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--cost", default="completion", choices=list(COSTS))
    ap.add_argument("--only", default=None, help="comma list of task names in the spec")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratify-seats", action="store_true",
                    help="draw independent groups evenly over the seats (the private-information conditions)")
    ap.add_argument("--seat-split", default="interp", choices=["interp", "balanced"])
    ap.add_argument("--strict-pool", action="store_true", help="grade a cell whose n_B exceeds the pool at levels 1-2 only")
    ap.add_argument("--vote-free", action="store_true", help="count-only vote on both sides (confidences dropped)")
    ap.add_argument("--bh-q", type=float, default=None)
    ap.add_argument("--bh-family", default="all", choices=list(FAMILIES))
    ap.add_argument("--task-rng", action="store_true", help="seed each task on its own")
    a = ap.parse_args(argv)
    if not Path(a.spec).exists():
        raise FileNotFoundError(f"grading spec {a.spec} missing: write it with the experiment driver's grade command")
    spec = json.loads(Path(a.spec).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shared = random.Random(a.seed)
    balanced = a.stratify_seats and a.seat_split == "balanced"
    flags = (("_strat", a.stratify_seats), ("_bal", balanced), ("_strict", a.strict_pool), ("_votefree", a.vote_free),
             ("_census", a.bh_family == "census"), ("_taskrng", a.task_rng))
    run_tag = a.cost + "".join(s for s, on in flags if on)
    done: dict = {}
    for name, task in spec.items():
        if a.only and name not in a.only.split(","):
            continue
        rng = random.Random(f"{a.seed}:{name}") if a.task_rng else shared
        free = a.vote_free or bool(task.get("vote_free"))
        pq = bool(task.get("per_question"))
        tag = a.cost + "".join(s for s, on in flags if on or (s == "_votefree" and free)) + ("_pq" if pq else "")
        seats = a.stratify_seats and (task["task"] == "hiddenbench" or bool(task.get("stratify_seats")))
        got = (per_question_task(name, task, a, rng, free, seats) if pq
               else standard_task(name, task, a, rng, free, seats, balanced))
        if got is None:
            continue
        rows = got["rows"]
        m_task = add_q(rows, "task", a.bh_family)
        rate = got["count"] / got["heads"] if got["heads"] else None
        notes = [f"pool answering rate {fq(rate)} (answering members per member paid for)",
                 "ub: one-sided 95% bound, mean + 1.645 SE; class: each contrast alone at 1.96 SE",
                 f"BH family {a.bh_family}; cells that stop early and capped cells (level 3) left out"]
        if pq:
            notes.append("per question: c1(q) per question, n_B per record, curves exact at every integer n")
        if seats:
            notes.append("independent groups drawn evenly over the seats"
                         + (" (exact at every n)" if (balanced or pq) else " (interpolated between multiples)"))
        if free:
            notes.append("count-only vote on both sides")
        if a.strict_pool or pq:
            notes.append("a cell whose n_B exceeds the pool is graded at levels 1-2 only")
        options = {"cost": a.cost, "stratify_seats": bool(seats),
                   "seat_split": ("balanced" if pq else a.seat_split) if seats else None,
                   "strict_pool": True if pq else a.strict_pool, "vote_free": free, "bh_q": a.bh_q,
                   "bh_family": a.bh_family, "bh_m": m_task, "seed": a.seed, "draws": a.draws, "task_rng": a.task_rng,
                   "per_question": pq, "own_family": bool(task.get("own_family"))}
        if pq:
            options["c1_per_question"] = got["c1_q"]
        res = {"c1": got["c1"], "rows": rows,
               "pool_mean": sum(len(v) for v in got["pool"].values()) / max(len(got["pool"]), 1),
               "notes": notes, "answer_rate": rate, "tag": tag, "options": options}
        done[name] = res
        print(write(out, name, tag, res, a.cost, None), end="", flush=True)
    run_done = [n for n in done if not spec[n].get("own_family")]
    m_run = add_q([r for n in run_done for r in done[n]["rows"]], "run", a.bh_family)
    for name, res in done.items():
        if spec[name].get("own_family"):
            m_own = add_q(res["rows"], "run", a.bh_family)
            res["options"].update(run_family=[name], bh_m={**res["options"]["bh_m"], **m_own})
        else:
            res["options"].update(run_family=run_done, bh_m={**res["options"]["bh_m"], **m_run})
            res["notes"].append(f"run-wide BH over {len(run_done)} task(s)")
        add_pass(res["rows"], a.bh_q)
        write(out, name, res["tag"], res, a.cost, a.bh_q)
    if a.bh_q is not None:
        lines = [f"# Benjamini-Hochberg at q <= {a.bh_q} ({run_tag}, family {a.bh_family})", "",
                 "| task | cell | root | kind | level | q task | q run |", "|---|---|---|---|---|---|---|"]
        for name, res in done.items():
            for r in res["rows"]:
                for kind, lv in (("vote", "2 (Y > V(n_B))"), ("truth", "3 (Y > T(n_B))")):
                    if r["bh_pass"][f"{kind}_task"] or r["bh_pass"][f"{kind}_run"]:
                        lines.append(f"| {name} | {r['cell']} | {r['root']} | {kind_of(r)} | {lv} | "
                                     f"{fq(r[f'q_{kind}_task'])} | {fq(r[f'q_{kind}_run'])} |")
        (out / f"bh_{run_tag}.md").write_text("\n".join(lines) + "\n")
    if not any(res["rows"] for res in done.values()):
        print(f"[grade] no gradable records for any cell of {a.spec}: run the lines first", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
