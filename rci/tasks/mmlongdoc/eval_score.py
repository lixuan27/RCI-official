"""Rule-based MMLongBench-Doc scoring, ported from the official eval/eval_score.py."""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from math import isclose

NOT_ANSWERABLE = "Not answerable"


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth, prediction, threshold=0.5):
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except (TypeError, ValueError):
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s):
    s = str(s).lower().strip()
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_exact_match(s):
    flag = False
    if "https://" in s:
        flag = True
    if s.endswith(".py") or s.endswith("ipynb"):
        flag = True
    if s.startswith("page"):
        flag = True
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        flag = True
    if "a.m." in s or "p.m." in s:
        flag = True
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        flag = True
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        flag = True
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        flag = True
    return flag


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def _as_list(v):
    if isinstance(v, str) and v.startswith("["):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    if isinstance(v, tuple):
        v = list(v)
    return v if isinstance(v, list) else [v]


def eval_score(gt, pred, answer_type):
    if answer_type == "Int":
        try:
            gt, pred = int(gt), int(float(pred))
        except (TypeError, ValueError, OverflowError):
            pred = ""
        score = (gt == pred)
    elif answer_type == "Float" and not isfloat(get_clean_string(str(gt))):
        gt, pred = get_clean_string(str(gt)), get_clean_string(str(pred))
        score = (gt == pred) if is_exact_match(gt) else anls_compute(gt, pred)
    elif answer_type == "Float":
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except (TypeError, ValueError):
            pred = ""
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
    elif answer_type in ["Str", "None"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = (gt == pred)
        else:
            score = anls_compute(gt, pred)
    else:
        gt, pred = _as_list(gt), _as_list(pred)
        if len(gt) != len(pred):
            score = 0.0
        elif not gt:
            score = 1.0
        else:
            gt = sorted([get_clean_string(a) for a in gt])
            pred = sorted([get_clean_string(a) for a in pred])
            if isfloat(gt[0]) or is_exact_match(gt[0]):
                score = ("-".join(gt) == "-".join(pred))
            else:
                score = min([anls_compute(gt_v, pred_v) for gt_v, pred_v in zip(gt, pred)])
    return float(score)


def eval_acc_and_f1(samples):
    """Upstream accuracy and F1 over samples carrying "score", "answer" and "pred"."""
    evaluated_samples = [sample for sample in samples if "score" in sample]
    if not evaluated_samples:
        return 0.0, 0.0
    acc = sum([sample["score"] for sample in evaluated_samples]) / len(evaluated_samples)
    try:
        answerable = [s for s in evaluated_samples if s["answer"] != NOT_ANSWERABLE]
        recall = sum([s["score"] for s in answerable]) / len(answerable)
        precision = sum([s["score"] for s in answerable]) / len(
            [s for s in evaluated_samples if s["pred"] != NOT_ANSWERABLE])
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0.0 else 0.0
    except ZeroDivisionError:
        f1 = 0.0
    return acc, f1


def breakdown(samples) -> dict:
    """The groups of upstream show_results, as a dict. evidence_pages / evidence_sources are lists here."""
    def row(sub):
        acc, f1 = eval_acc_and_f1(sub)
        return {"acc": acc, "f1": f1, "n": len(sub)}

    out = {"overall": row(samples),
           "single_page": row([s for s in samples if len(s["evidence_pages"]) == 1]),
           "cross_page": row([s for s in samples if len(s["evidence_pages"]) != 1 and s["answer"] != NOT_ANSWERABLE]),
           "unanswerable": row([s for s in samples if s["answer"] == NOT_ANSWERABLE])}
    by_src, by_type = defaultdict(list), defaultdict(list)
    for s in samples:
        for src in s["evidence_sources"]:
            by_src[src].append(s)
        by_type[s["doc_type"]].append(s)
    out["evidence_source"] = {k: row(v) for k, v in sorted(by_src.items())}
    out["doc_type"] = {k: row(v) for k, v in sorted(by_type.items())}
    return out
