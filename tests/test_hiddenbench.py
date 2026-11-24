import asyncio
import json

import pytest

from rci.agents.runner import GroupRun, RunConfig
from rci.protocols.independent import IndependentAgents
from rci.registry import make_task
from rci.tasks.hiddenbench.task import norm, share
from fake_llm import ScriptedLLM

ROW = {"id": 1, "name": "t", "description": "Pick a route.", "shared_information": ["West looks fine."],
       "hidden_information": ["h0", "h1", "h2", "h3"], "possible_answers": ["West City", "East Town"], "correct_answer": "East Town"}


def task(tmp_path, **kw):
    (tmp_path / "benchmark.json").write_text(json.dumps([ROW]))
    return make_task("hiddenbench", root=str(tmp_path), **kw)


def test_each_seat_holds_its_share_of_the_private_facts():
    assert share(["a", "b", "c", "d"], 1, 4) == ["b"]
    assert share(["a", "b", "c", "d"], 1, 2) == ["b", "d"]
    assert share(["a", "b"], 3, 4) == []


def test_profiles_set_what_a_member_sees(tmp_path):
    ex = task(tmp_path).examples()[0]
    assert ex["id"] == "hb_01"
    hidden = task(tmp_path, profile="hidden").task_prompt(ex, 2, 4, None)
    assert "h2" in hidden and "h0" not in hidden and "West looks fine." in hidden
    full = task(tmp_path, profile="full").task_prompt(ex, 2, 4, None)
    assert all(h in full for h in ROW["hidden_information"])
    assert "h2" not in task(tmp_path, profile="shared").task_prompt(ex, 2, 4, None)


def test_an_unknown_profile_is_refused(tmp_path):
    with pytest.raises(ValueError):
        task(tmp_path, profile="secret")


def test_scoring_ignores_case_and_punctuation(tmp_path):
    t = task(tmp_path)
    ex = t.examples()[0]
    assert t.score_agent(ex, {"answer": " east town. "}) == 1.0
    assert t.score_agent(ex, {"answer": "West City"}) == 0.0
    assert t.score_agent(ex, None) == 0.0 and norm(None) == ""


def test_a_group_run_scores_and_keeps_predictions_for_the_vote(tmp_path):
    t = task(tmp_path, confidence=True)
    ex = t.examples()[0]

    def policy(messages, tools, route_key, tag):
        i = int(route_key.rsplit("a", 1)[1])
        return [("submit_answer", {"answer": "East Town" if i == 3 else "West City", "confidence": 0.4})]
    rec = asyncio.run(GroupRun(t, ex, IndependentAgents(), RunConfig(n_agents=4, max_actions=4, max_steps=6, seed=0, summarize=False), ScriptedLLM(policy)).run())
    assert rec["score"]["score"] == 1.0 and rec["score"]["per_agent"] == [0.0, 0.0, 0.0, 1.0]
    assert rec["score"]["preds"] == ["west city", "west city", "west city", "east town"]


def test_the_hint_is_off_by_default(tmp_path):
    ex = task(tmp_path).examples()[0]
    assert "Before you decide" not in task(tmp_path).task_prompt(ex, 0, 4, None)
    assert "Before you decide" in task(tmp_path, hint=True).task_prompt(ex, 0, 4, None)


def test_relayed_holds_everything_full_does_but_the_others_facts_come_as_what_they_said(tmp_path):
    ex = task(tmp_path).examples()[0]
    t = task(tmp_path, profile="relayed")
    p = t.task_prompt(ex, 2, 4, None)
    own, told = p.split("What the other participants have told you:")
    assert "h2" in own and not any(h in own for h in ("h0", "h1", "h3"))
    assert all(f"Participant {k}: h{k}" in told for k in (0, 1, 3)) and "Participant 2" not in told
    assert t.facts(ex, 2, 4) == task(tmp_path, profile="hidden").facts(ex, 2, 4)
    assert "told you" not in task(tmp_path, profile="full").task_prompt(ex, 2, 4, None)
