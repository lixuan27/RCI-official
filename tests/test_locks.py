import os
import time

from rci import run as run_eval


def test_lock_exclusive_reclaim_and_stale(tmp_path):
    p = tmp_path / "N4" / "s0" / "q1.json.gz"
    assert run_eval.try_lock(p)
    assert run_eval.try_lock(p)
    lock = p.with_suffix(p.suffix + ".lock")
    lock.write_text("someone-else")
    assert not run_eval.try_lock(p)
    old = time.time() - run_eval.LOCK_STALE_S - 5
    os.utime(lock, (old, old))
    assert run_eval.try_lock(p)
    run_eval.release_lock(p)
    assert not lock.exists()
