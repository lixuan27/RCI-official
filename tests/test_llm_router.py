from rci.agents.llm import Router


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_new_keys_spread_evenly_and_keys_stay_put():
    r = Router(4, clock=Clock())
    first = [r.pick(f"run{i}a{j}") for i in range(4) for j in range(4)]
    assert sorted(first.count(k) for k in range(4)) == [4, 4, 4, 4]
    assert [r.pick(f"run{i}a{j}") for i in range(4) for j in range(4)] == first


def test_in_flight_requests_steer_new_keys_away():
    r = Router(2, clock=Clock())
    a = r.pick("a")
    r.inflight[a] += 5
    assert r.pick("b") != a and r.pick("c") != a


def test_stale_keys_stop_counting_and_old_keys_are_forgotten():
    c = Clock()
    r = Router(2, clock=c)
    for k in "abcd":
        r.pick(k)
    on0 = [k for k in "abcd" if r.keys[k][0] == 0]
    c.t = Router.LIVE_S + 1
    for k in on0:
        r.pick(k)
    assert r.pick("e") == 1
    r.MAX_KEYS = 3
    r.pick("f")
    assert len(r.keys) == 3 and list(r.keys)[-1] == "f"


def test_single_replica_and_keyless_calls():
    assert Router(1).pick("x") == 0
    r = Router(3, clock=Clock())
    assert all(0 <= r.pick(None) < 3 for _ in range(10)) and not r.keys


def test_a_failed_replica_gets_no_new_keys_until_it_cools_down_and_keys_move_on_failover():
    c = Clock()
    r = Router(2, clock=c)
    r.fail(0)
    assert all(r.pick(f"k{i}") == 1 for i in range(5))
    c.t = Router.COOLDOWN_S + 1
    assert r.pick("fresh") == 0
    r.rebind("k0", 0)
    assert r.pick("k0") == 0
