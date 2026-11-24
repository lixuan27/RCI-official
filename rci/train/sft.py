"""Full-parameter supervised fine-tuning on token rows with loss spans (the recursion generations)."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path


def lr_at(step: int, total: int, base: float, warmup: float) -> float:
    """Linear warm-up over the first `warmup` share of the steps, then a cosine decay to 0."""
    w = max(1, int(total * warmup))
    if step < w:
        return base * (step + 1) / w
    t = (step - w) / max(1, total - w)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def batches(n: int, size: int, epochs: int, seed: int) -> list[list[int]]:
    """Index batches of a shuffled pass per epoch; a pure function of the arguments, so every rank agrees."""
    out = []
    for e in range(epochs):
        order = list(range(n))
        random.Random(f"{seed}/{e}").shuffle(order)
        out += [order[i:i + size] for i in range(0, n, size)]
    return out


def resume_point(out: Path, total: int) -> tuple:
    """(the step to go on from, the partial export holding it), or (0, None) for a fresh run."""
    part = out / "partial"
    done = sorted(int(p.name[len("step_"):]) for p in part.glob("step_*") if (p / "config.json").exists())
    done = [n for n in done if n <= total]
    return (done[-1], part / f"step_{done[-1]:06d}") if done else (0, None)


def export(pol, a, out: Path, step: int, rank: int, final: bool = False) -> None:
    """Write a servable export: <out>/hf when final, else <out>/partial/step_<n> (the previous one is removed)."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    from rci.train.hf_export import write_hf

    sd = get_model_state_dict(pol, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank != 0:
        return
    dest = out / "hf" if final else out / "partial" / f"step_{step:06d}"
    write_hf(sd, Path(a.model), dest, {"step": step, "run_id": out.name, "source": "sft"})
    if final and getattr(a, "master", False):
        write_hf(sd, Path(a.model), out / "master", {"step": step, "run_id": out.name, "source": "sft"}, "float32")
    for old in (out / "partial").glob("step_*"):
        if final or old.name != dest.name:
            shutil.rmtree(old, ignore_errors=True)
    print("exported", step, flush=True)


def main(argv=None) -> int:
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from rci.train.loss import TrainItem, balance, pad_right
    from rci.train.policy import build_policy, set_checkpointing

    ap = argparse.ArgumentParser(description="Supervised fine-tuning on JSONL rows {ids, spans, w}; run with torchrun. "
                                             "The final weights are exported to <out>/hf.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=None, help="optimiser steps, overriding --epochs (equal compute across seats)")
    ap.add_argument("--save-every", type=int, default=0, help="partial export every this many steps; a rerun resumes from it")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--pad-multiple", type=int, default=1024)
    ap.add_argument("--ckpt-min-len", type=int, default=4096)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--master", action="store_true", help="also write the full-precision weights to <out>/master")
    ap.add_argument("--offload-activations", action="store_true",
                    help="keep the tensors saved for backward in pinned CPU memory (needed for 27B rows of 30-64k tokens)")
    ap.add_argument("--load-group", type=int, default=0,
                    help="ranks that load the fp32 model on the CPU at once (0 = all)")
    a = ap.parse_args(argv)

    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=dev, timeout=timedelta(minutes=30))
    mesh = init_device_mesh("cuda", (world,))
    torch.manual_seed(a.seed)
    rows = [json.loads(line) for line in open(a.data)]
    rows = rows[:a.max_seqs] if a.max_seqs else rows
    items = [TrainItem(r["ids"], r["spans"][0][0], [], 1.0, weight=float(r.get("w", 1.0)), spans=r["spans"]) for r in rows]
    plan = batches(len(items), a.batch, math.ceil(a.steps * a.batch / len(items)) if a.steps else a.epochs, a.seed)
    plan = plan[:a.steps] if a.steps else plan
    scale = a.batch * sum(it.n * it.weight for it in items) / len(items)
    out = Path(a.out)
    start, source = resume_point(out, len(plan))
    group = a.load_group if a.load_group > 0 else world
    pol = None
    for g0 in range(0, world, group):
        if g0 <= rank < g0 + group:
            pol = build_policy(str(source) if source else a.model, mesh)
            import gc
            gc.collect()
        dist.barrier()
    params = [p for p in pol.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({"run_id": out.name, "model_path": a.model, "sft": vars(a), "n_seqs": len(items)}, indent=1))
    log = open(out / "metrics.jsonl", "a") if rank == 0 else None
    dummy = torch.zeros((1, a.pad_multiple), dtype=torch.long, device=dev)
    pol.train()
    t0 = time.time()
    for step, idx in enumerate(plan):
        if step < start:
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step, len(plan), a.lr, a.warmup)
        batch = [items[i] for i in idx]
        n_tok = sum(it.n for it in batch)
        total = torch.zeros(1, device=dev, dtype=torch.float64)
        for it in balance(batch, world, a.pad_multiple)[rank]:
            if it is None:
                (pol(dummy, 1, 1).sum() * 0.0).backward()
                continue
            ids = torch.tensor([pad_right(it.ids, a.pad_multiple)], dtype=torch.long, device=dev)
            set_checkpointing(pol, ids.shape[1] >= a.ckpt_min_len)
            with (torch.autograd.graph.save_on_cpu(pin_memory=True) if a.offload_activations else contextlib.nullcontext()):
                lp = pol(ids, torch.tensor(it.positions(), dtype=torch.long, device=dev))
            loss = -lp.sum() * it.weight / scale
            loss.backward()
            total += loss.detach().double()
        gnorm = torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
        gnorm = float(gnorm.full_tensor() if hasattr(gnorm, "full_tensor") else gnorm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(total)
        if rank == 0:
            rec = {"step": step + 1, "of": len(plan), "nll": float(total.item()), "lr": opt.param_groups[0]["lr"], "gnorm": gnorm,
                   "n_tok": n_tok, "elapsed_s": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if a.save_every and (step + 1) % a.save_every == 0 and step + 1 < len(plan):
            export(pol, a, out, step + 1, rank)
            dist.barrier()
    export(pol, a, out, len(plan), rank, final=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
