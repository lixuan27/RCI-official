import pytest

from rci.train.credit import agent_credits, citation_flow


def ca_rec(finals, messages, score):
    return {"task": "audit", "score": {"score": score}, "finals": finals, "method_state": {"messages": messages},
            "agents": [{"id": f["agent"], "failing": f.get("failing", False)} for f in finals]}


EX = {"contradictions": [{"edited": 5, "partner": 1}, {"edited": 9, "partner": 3}]}


def test_broadcast_gives_everyone_R():
    rec = ca_rec([{"agent": 0, "final": None}, {"agent": 1, "final": {"pairs": [{"a": 1, "b": 5}]}}], [], 0.5)
    assert agent_credits(rec, EX, "team") == {0: 0.5, 1: 0.5}


def test_citation_flow_rewards_cited_author_and_conserves():
    msgs = [{"id": 0, "author": 0, "msg_refs": []},
            {"id": 1, "author": 1, "msg_refs": [0]}]
    finals = [{"agent": 0, "final": {"pairs": [], "refs": []}},
              {"agent": 1, "final": {"pairs": [{"a": 1, "b": 5}], "refs": ["Message1"]}}]
    flow = citation_flow({1: 0.5}, finals, msgs, beta=0.5)
    assert sum(flow.values()) == pytest.approx(0.5)
    assert flow[0] == pytest.approx(0.125) and flow[1] == pytest.approx(0.375)
    cred = agent_credits(ca_rec(finals, msgs, 0.5), EX, "shapley_flow", beta=0.5, lam=0.0)
    assert cred[0] == pytest.approx(2 * 0.125) and cred[1] == pytest.approx(2 * 0.375)
    assert sum(cred.values()) / 2 == pytest.approx(0.5)


def test_shared_report_is_split_and_failing_agent_excluded():
    finals = [{"agent": 0, "failing": True, "final": None},
              {"agent": 1, "final": {"pairs": [{"a": 1, "b": 5}]}},
              {"agent": 2, "final": {"pairs": [{"a": 5, "b": 1}, {"a": 9, "b": 3}]}}]
    cred = agent_credits(ca_rec(finals, [], 1.0), EX, "shapley_flow", lam=0.0)
    assert set(cred) == {1, 2}
    assert cred[1] == pytest.approx(2 * 0.25) and cred[2] == pytest.approx(2 * 0.75)


def test_shapley_splits_each_finding_among_its_finders_and_ignores_citations():
    msgs = [{"id": 0, "author": 0, "msg_refs": []}]
    finals = [{"agent": 0, "final": {"pairs": [{"a": 1, "b": 5}], "refs": []}},
              {"agent": 1, "final": {"pairs": [{"a": 5, "b": 1}], "refs": ["Message0"]}},
              {"agent": 2, "final": {"pairs": [{"a": 9, "b": 3}], "refs": []}}]
    cred = agent_credits(ca_rec(finals, msgs, 1.0), EX, "shapley", lam=0.0)
    assert cred == pytest.approx({0: 3 * 0.25, 1: 3 * 0.25, 2: 3 * 0.5})
    assert sum(cred.values()) / 3 == pytest.approx(1.0)
    cit = agent_credits(ca_rec(finals, msgs, 1.0), EX, "shapley_flow", lam=0.0, beta=0.5)
    assert cit[0] > cred[0] and cit[1] < cred[1]


def test_shapley_on_pass_at_n_splits_among_correct_agents():
    rec = {"task": "bcp", "score": {"score": 1.0, "per_agent": [1.0, 0.0, 1.0, 0.0]},
           "finals": [{"agent": i, "final": {"answer": "x"}} for i in range(4)], "method_state": {},
           "agents": [{"id": i} for i in range(4)]}
    cred = agent_credits(rec, {}, "shapley", lam=0.0)
    assert cred == pytest.approx({0: 2.0, 1: 0.0, 2: 2.0, 3: 0.0})


def _bcp_rec(answers_conf_per, msgs=(), refs=None):
    finals = [{"agent": i, "final": ({"answer": "x", "confidence": b, "refs": (refs or {}).get(i, [])} if b != "none" else None)}
              for i, (b, _) in enumerate(answers_conf_per)]
    per = [p for _, p in answers_conf_per]
    return {"task": "bcp", "score": {"score": max(p for p in per if p is not None), "per_agent": per},
            "finals": finals, "method_state": {"messages": list(msgs)}, "agents": [{"id": i} for i in range(len(per))]}


def test_flow_reduces_to_the_closed_forms():
    from rci.train.credit import game_values

    finals = [{"agent": 0, "final": {"pairs": [{"a": 1, "b": 5}]}},
              {"agent": 1, "final": {"pairs": [{"a": 5, "b": 1}, {"a": 9, "b": 3}]}}]
    R, sh, _ = game_values(ca_rec(finals, [], 1.0), EX)
    assert R == 1.0 and sh == pytest.approx({0: 0.25, 1: 0.75})
    rec = {"task": "bcp", "score": {"score": 1.0, "per_agent": [1.0, 0.0, 1.0]}, "method_state": {},
           "finals": [{"agent": i, "final": {"answer": "x"}} for i in range(3)], "agents": [{"id": i} for i in range(3)]}
    assert game_values(rec, {})[1] == pytest.approx({0: 0.5, 1: 0.0, 2: 0.5})
    R, sh, _ = game_values(_bcp_rec([(0.6, 1.0), (0.2, 0.0), (0.0, 0.0)]), {})
    assert R == 1.0 and sh == pytest.approx({0: 1.0, 1: 0.0, 2: 0.0})


def test_flow_pays_every_positive_share_at_its_holders_confidence():
    msgs = [{"id": 0, "author": 3, "msg_refs": []}]
    rec = _bcp_rec([(0.8, 1.0), (0.4, 1.0), (0.2, 0.0), (0.0, 0.0)], msgs, refs={1: ["Message0"]})
    from rci.train.credit import game_values
    R, sh, confs = game_values(rec, {})
    assert R == 1.0 and sh == pytest.approx({0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0})
    cred = agent_credits(rec, {}, "shapley_flow", lam=0.0, beta=0.5)
    assert cred[3] == pytest.approx(4 * 0.4 * sh[1]) and cred[1] == pytest.approx(4 * 0.6 * sh[1])
    assert sum(cred.values()) / 4 == pytest.approx(R)


def test_verified_task_credit_is_the_coverage_game():
    from rci.train.credit import agent_credits, game_values

    def rec(per, confs):
        return {"task": "fate", "score": {"score": max(per), "per_agent": per}, "method_state": {"messages": []},
                "agents": [{"id": i} for i in range(len(per))],
                "finals": [{"agent": i, "final": {"proof": f"p{i}", "confidence": b}} for i, b in enumerate(confs)]}
    R, shares, confs = game_values(rec([0.0, 1.0, 1.0, 0.0], [0.8, 0.2, 0.4, 0.8]), {}, "vote_stake")
    assert R == 1.0 and confs == {0: 0.8, 1: 0.2, 2: 0.4, 3: 0.8}
    assert all(abs(shares[i] - v) < 1e-9 for i, v in {0: 0.0, 1: 0.5, 2: 0.5, 3: 0.0}.items())
    R, shares, _ = game_values(rec([0.0, 0.0], [0.8, 0.8]), {}, "auction")
    assert R == 0.0 and all(abs(v) < 1e-9 for v in shares.values()) and set(shares) == {0, 1}
    c = agent_credits(rec([0.0, 1.0, 1.0, 0.0], [0.8, 0.2, 0.4, 0.8]), {}, mode="shapley", lam=0.25, commit="vote_stake")
    assert abs(sum(c.values()) / 4 - 1.0) < 1e-9 and c[1] == c[2] > c[0] == c[3]
