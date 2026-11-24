"""Sharded DCP checkpoints (model + optimizer) with atomic publish and keep-N pruning."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import torch.distributed as dist

STEP_RE = re.compile(r"^step_(\d{6})$")


def latest(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    done = [p for p in root.iterdir() if STEP_RE.match(p.name) and (p / "meta.json").exists()]
    return max(done, key=lambda p: int(STEP_RE.match(p.name).group(1))) if done else None


def save(root: Path, step: int, model, optim, meta: dict, keep: int) -> Path:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    rank = dist.get_rank()
    tmp, final = root / f".tmp_step_{step:06d}", root / f"step_{step:06d}"
    if rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
    dist.barrier()
    msd, osd = get_state_dict(model, optim)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(tmp))
    dist.barrier()
    if rank == 0:
        _publish(root, tmp, final, {**meta, "step": step}, keep)
    dist.barrier()
    return final


def _publish(root: Path, tmp: Path, final: Path, meta: dict, keep: int) -> None:
    (tmp / "meta.json").write_text(json.dumps(meta))
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)
    done = sorted((p for p in root.iterdir() if STEP_RE.match(p.name)), key=lambda p: p.name)
    for old in done[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


class AsyncSaver:
    def __init__(self, root: Path, keep: int, group):
        self.root, self.keep, self.group = root, keep, group
        self.pending: dict | None = None
        self.last_write_s: float | None = None
        self._lock = threading.Lock()

    def save(self, step: int, model, optim, meta: dict) -> float:
        """Stage `step` and start writing it; returns the seconds the caller was blocked (staging)."""
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict

        t0 = time.time()
        self.wait()
        tmp, final = self.root / f".tmp_step_{step:06d}", self.root / f"step_{step:06d}"
        if dist.get_rank() == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
        dist.barrier(group=self.group)
        msd, osd = get_state_dict(model, optim)
        res = dcp.async_save({"model": msd, "optim": osd}, checkpoint_id=str(tmp), process_group=self.group)
        staging = getattr(res, "staging_completion", None)
        if staging is not None:
            staging.result()
        e = {"fut": getattr(res, "upload_completion", res), "tmp": tmp, "final": final, "meta": {**meta, "step": step},
             "t": time.time(), "done": False}
        if dist.get_rank() == 0:
            e["fut"].add_done_callback(lambda _f, e=e: self._finish(e))
        self.pending = e
        return time.time() - t0

    def _finish(self, e: dict) -> None:
        with self._lock:
            if e["done"]:
                return
            e["done"] = True
            err = e["fut"].exception()
            if err is not None:
                print(f"[ckpt] background save of step {e['meta']['step']} failed: {err!r}", file=sys.stderr, flush=True)
                return
            _publish(self.root, e["tmp"], e["final"], e["meta"], self.keep)
            self.last_write_s = time.time() - e["t"]

    def poll(self) -> None:
        """Forget the in-flight save once it has finished (non-blocking; rank 0 published it already)."""
        if self.pending is not None and self.pending["fut"].done():
            self.wait()

    def wait(self) -> None:
        e, self.pending = self.pending, None
        if e is None:
            return
        try:
            e["fut"].result()
        except Exception:
            pass
        if dist.get_rank() == 0:
            self._finish(e)


def load(path: Path, model, optim) -> dict:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    msd, osd = get_state_dict(model, optim)
    state = {"model": msd, "optim": osd}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, optim, model_state_dict=state["model"], optim_state_dict=state["optim"])
    return json.loads((path / "meta.json").read_text())
