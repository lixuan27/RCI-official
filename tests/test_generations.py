import json
import random

import pytest

from rci.recursion import contrasts
from rci.tasks.splitbench import split


def record(q, seed, answers, tokens=400):
    per = [1.0 if a == "A" else 0.0 for a in answers]
    return {"task": "hiddenbench", "example_id": q, "config": {"seed": seed},
            "score": {"score": max(per), "per_agent": per, "preds": answers},
            "compute": {"completion_tokens": tokens, "total_tokens": tokens * 3},
            "agents": [{"id": i} for i in range(len(answers))],
            "finals": [{"agent": i, "final": {"answer": a, "confidence": 0.6}} for i, a in enumerate(answers)]}


def write_cell(root, protocol, seeds, questions, p_right, tokens, rng):
    for s in seeds:
        d = root / protocol / "N4" / f"s{s}"
        d.mkdir(parents=True, exist_ok=True)
        for q in questions:
            answers = ["A" if rng.random() < p_right else "B" for _ in range(4)]
            (d / f"{q}.json").write_text(json.dumps(record(q, s, answers, tokens)))


@pytest.fixture
def chain(tmp_path):
    rng = random.Random(0)
    qs = [f"hb_{1000 + i}" for i in range(24)]
    write_cell(tmp_path / "pool", "independent", range(8), qs, 0.3, 400, rng)
    skill = {"base": 0.25, "g1a": 0.6, "g1b": 0.6, "g2a": 0.9, "g2b": 0.9, "ca": 0.55, "cb": 0.55}
    for m, p in skill.items():
        write_cell(tmp_path / "eval" / m, "debate_r3", range(6, 10), qs, p, 1600, rng)
    spec = {"base_pool": {"root": str(tmp_path / "pool"), "protocol": "independent", "n": 4},
            "generations": {"G0": ["base"], "G1": ["g1a", "g1b"], "G2": ["g2a", "g2b"], "control": ["ca", "cb"]},
            "eval_root": str(tmp_path / "eval" / "{model}"), "protocol": "debate_r3", "n": 4, "seeds": [6, 7, 8, 9],
            "multipliers": {"step": 1.96, "control": 2.24}, "draws": 200}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_contrasts_read_steps_and_the_control_from_records(chain, tmp_path):
    contrasts.main([str(chain), "--out", str(tmp_path / "out")])
    res = json.loads((tmp_path / "out" / "contrasts.json").read_text())
    c = res["contrasts"]
    assert c["G0_to_G1"]["mean"] > 0 and c["G1_to_G2"]["mean"] > 0 and c["G2_minus_control"]["mean"] > 0
    assert all(c[k]["se"] > 0 for k in ("G0_to_G1", "G1_to_G2", "G2_minus_control"))


def test_holm_multipliers_are_the_normal_quantiles():
    from statistics import NormalDist
    q = NormalDist().inv_cdf
    assert q(1 - 0.025 / 3) == pytest.approx(2.39, abs=0.01)
    assert q(1 - 0.025 / 2) == pytest.approx(2.24, abs=0.01)
    assert q(1 - 0.025) == pytest.approx(1.96, abs=0.01)


def test_split_is_a_deterministic_hash_of_the_ids():
    ids = [f"hb_{1000 + i}" for i in range(697)]
    train, held = split.split_ids(ids)
    assert len(held) == 196 and len(train) == 501 and not set(train) & set(held)
    assert split.split_ids(list(reversed(ids))) == (train, held)
    assert split.split_ids(ids, salt="other")[1] != held


def test_seal_moves_the_writer_key_out_of_the_task_file(tmp_path):
    rows = [{"id": 1000 + i, "description": "d", "shared_information": ["s"], "hidden_information": ["h"],
             "possible_answers": ["A", "B", "C"], "writer_answer": "A", "decoy": "B", "explanation": "e",
             "full_answer": "A" if i else "C", "full_agree": 1.0, "matches_writer": bool(i)} for i in range(3)]
    (tmp_path / "benchmark.json").write_text(json.dumps(rows))
    got = split.seal(tmp_path)
    assert got == {"sealed": 3, "key_differs_from_full_answer": 1}
    sealed = json.loads((tmp_path / "benchmark.json").read_text())
    assert all("writer_answer" not in r and "decoy" not in r and "explanation" not in r for r in sealed)
    assert [r["correct_answer"] for r in sealed] == ["C", "A", "A"]
    key = json.loads((tmp_path / "writer_key.json").read_text())
    assert key["hb_1000"]["writer_answer"] == "A"
    assert split.seal(tmp_path)["already_sealed"]
