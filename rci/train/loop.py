"""On-policy training of members that share one policy, with vLLM rollouts and in-place weight sync."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

STOP = False
JUDGE_KEYS = ("judge_url", "judge_model")
JUDGED = ("browsecomp", "mmlongdoc")


def _on_term(signum, frame):
    global STOP
    STOP = True


def log(rank: int, *a) -> None:
    if rank == 0:
        print(f"[rank0 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_streams(cfg) -> list[tuple[str, dict, int, int, float]]:
    judge = {k: cfg.task_kwargs[k] for k in JUDGE_KEYS if k in cfg.task_kwargs}
    primary_kw = cfg.task_kwargs if cfg.task in JUDGED else {k: v for k, v in cfg.task_kwargs.items()
                                                              if k not in JUDGE_KEYS}
    pb = cfg.beta if cfg.primary_beta is None else cfg.primary_beta
    out = [(cfg.task, primary_kw, cfg.max_actions, cfg.max_steps, pb)]
    for m in cfg.mix:
        kw = {**(judge if m["task"] in JUDGED else {}), **(m.get("task_kwargs") or {})}
        out.append((m["task"], kw, m.get("max_actions", cfg.max_actions), m.get("max_steps", cfg.max_steps),
                    float(m.get("beta", cfg.beta))))
    return out


class Rollouts:
    """Rank-0 side: schedules question jobs on a spawn process pool and reads the rollout files back."""

    def __init__(self, cfg, out: Path):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        from rci.registry import make_task
        from rci.train.data import MixedSource, make_source

        self.cfg, self.out = cfg, out
        self.streams = build_streams(cfg)
        primary_kw = self.streams[0][1]
        task = make_task(cfg.task, **primary_kw)
        frozen, sc = out / "source.json", dict(cfg.source)
        if cfg.val_projects:
            sc["exclude"] = list(cfg.val_projects)
        if frozen.exists():
            d = json.loads(frozen.read_text())
            d = d["streams"][0] if d.get("kind") == "mixed" else d
            if d.get("items") is not None:
                sc["projects"], sc["kind"] = d["items"], d["kind"]
        self.source = make_source(cfg.task, task, primary_kw, sc, cfg.seed)
        if cfg.mix:
            extra = [make_source(t, make_task(t, **kw), kw, dict(m.get("source") or {}), cfg.seed)
                     for (t, kw, _, _, _), m in zip(self.streams[1:], cfg.mix)]
            self.source = MixedSource([self.source] + extra, [cfg.primary_weight] + [int(m.get("weight", 1))
                                                                                  for m in cfg.mix])
        if not frozen.exists():
            frozen.write_text(json.dumps(self.source.describe()))
        self.pool = ProcessPoolExecutor(max_workers=cfg.rollout_workers, mp_context=mp.get_context("spawn"))
        self.val_examples = None
        if cfg.val_every and cfg.val_projects:
            vsrc = make_source(cfg.task, task, primary_kw, {**dict(cfg.source), "projects": list(cfg.val_projects),
                                                                 "exclude": None}, cfg.seed)
            self.val_examples = []
            for i in range(len(vsrc)):
                ex = dict(vsrc.example(i))
                ex["id"] = f"val_{ex['id']}"
                self.val_examples.append(ex)

    def _job(self, ex: dict, path: Path, version: str, step: int) -> dict:
        c = self.cfg
        task, task_kwargs, max_actions, max_steps, beta = self.streams[ex.get("_stream", 0)]
        return {"task": task, "task_kwargs": task_kwargs, "example": ex, "path": str(path), "version": version,
                "step": step, "seed": c.seed, "protocol": c.protocol, "protocol_kwargs": c.protocol_kwargs,
                "n_agents": c.n_agents, "trainees": c.trainees,
                "credit": c.credit, "beta": beta, "lam": c.lam,
                "k": c.k, "max_actions": max_actions, "max_steps": max_steps, "gen_limit": c.gen_limit,
                "enable_thinking": c.enable_thinking, "summarize": c.summarize,
                "max_samples_per_agent": c.max_samples_per_agent, "include_summaries": c.include_summaries,
                "merge_prefix": c.merge_prefix, "l_max": c.l_max, "adv_mode": c.adv_mode,
                "commit": c.commit, "silent_floor": c.silent_floor,
                "policy_urls": c.policy_urls, "peer_urls": c.peer_urls, "served_policy": c.served_policy,
                "served_peer": c.served_peer, "llm_concurrency": c.llm_concurrency}

    def submit(self, step: int, version: str, allowed_version: str | None, examples: list[dict] | None = None,
               subdir: str | None = None, k: int | None = None) -> dict:
        from rci.train.rollout_io import qfile, reusable, run_question

        q = self.cfg.questions_per_step
        step_dir = self.out / "rollouts" / (subdir or f"step_{step:06d}")
        step_dir.mkdir(parents=True, exist_ok=True)
        if examples is None:
            examples = []
            for i in range((step - 1) * q, step * q):
                ex = dict(self.source.example(i))
                ex["id"] = f"{ex['id']}@{i}"
                examples.append(ex)
        files, futs, reused = [], {}, 0
        for ex in examples:
            path = qfile(step_dir, ex["id"])
            if reusable(path, allowed_version):
                reused += 1
                files.append(path)
                continue
            job = self._job(ex, path, version, step)
            if k is not None:
                job["k"] = k
            futs[self.pool.submit(run_question, job)] = path
        return {"step": step, "version": version, "files": files, "futs": futs, "reused": reused,
                "t0": time.time(), "result": None}

    def wait(self, h: dict):
        """Block until the handle's rollouts are done: (samples, summaries, info), or None if interrupted."""
        from concurrent.futures import FIRST_COMPLETED, wait

        from rci.train.rollout_io import read_question

        if h["result"] is not None:
            return h["result"]
        failed = []
        pending = set(h["futs"])
        while pending:
            done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    r = f.result()
                except Exception:
                    if STOP:
                        return None
                    raise
                if r.get("ok"):
                    h["files"].append(h["futs"][f])
                else:
                    failed.append(r)
                    print(f"[rollout] FAILED {r.get('question')}: {r.get('error')}\n{r.get('trace', '')}", flush=True)
            if STOP:
                for f in pending:
                    f.cancel()
                for p in list(getattr(self.pool, "_processes", {}).values()):
                    p.terminate()
                return None
        samples, summaries = [], []
        for p in h["files"]:
            hd, ss = read_question(p)
            samples += ss
            summaries += hd["summaries"]
        h["result"] = (samples, summaries, {"reused": h["reused"], "failed": len(failed), "files": len(h["files"]),
                                            "version": h["version"], "rollout_wall_s": round(time.time() - h["t0"], 1)})
        return h["result"]

    def collect(self, step: int, version: str, allowed_version: str | None):
        """Rollouts for update `step` (sampled at weight version `version`)."""
        return self.wait(self.submit(step, version, allowed_version))

    def submit_val(self, step: int, version: str) -> dict | None:
        if self.val_examples is None:
            return None
        return self.submit(step, version, None, examples=[dict(e) for e in self.val_examples],
                           subdir=f"val_{step:06d}", k=self.cfg.val_k)

    def val_metrics(self, h: dict | None) -> dict | None:
        """Mean group reward, per-member reward and distinct audit pairs of a validation handle."""
        got = self.wait(h) if h is not None else None
        if got is None:
            return None
        _, summ, info = got
        scores = [m["score"] for m in summ if m.get("score") is not None]
        return {"val_reward_mean": _mean(scores), "val_per_agent_mean": _mean([m.get("per_agent_mean") for m in summ]),
                "val_distinct_pairs": _mean([m.get("distinct_pairs") for m in summ]), "val_n": len(scores),
                "val_failed": info["failed"], "val_s": info["rollout_wall_s"]}

    def validate(self, step: int, version: str) -> dict | None:
        """Synchronous held-out validation (no rollouts in flight)."""
        return self.val_metrics(self.submit_val(step, version))

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def main(argv=None) -> int:
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from rci.train import ckpt
    from rci.train.config import load_config
    from rci.train.loss import balance, items_from_samples, pad_right, padded_len, split_minibatches, token_pg
    from rci.train.policy import build_policy, make_weight_source, set_checkpointing
    from rci.train.sync import VLLMControl, WeightPusher

    ap = argparse.ArgumentParser(description="Train members that share one policy on group rollouts under a credit rule "
                                             "(run with torchrun).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], help="key=value (YAML value), dotted keys allowed")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    cfg.validate()

    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=dev, timeout=timedelta(minutes=30))
    gloo = dist.new_group(backend="gloo", timeout=timedelta(hours=12))
    ck_group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
    signal.signal(signal.SIGTERM, _on_term)

    out = Path(cfg.out_root) / cfg.run_id
    if rank == 0:
        (out / "ckpt").mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1, default=str))
    dist.barrier()
    log(rank, f"world={world} run={cfg.run_id} model={cfg.model_path} trainees={cfg.trainees} credit={cfg.credit}")

    torch.manual_seed(cfg.seed)
    mesh = init_device_mesh("cuda", (world,))
    t0 = time.time()
    pol = build_policy(cfg.model_path, mesh, chunk=cfg.lm_chunk)
    params = [p for p in pol.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay)
    log(rank, f"policy built in {time.time() - t0:.0f}s; trainable={sum(p.numel() for p in params)} "
              f"mem={torch.cuda.memory_allocated() / 1e9:.1f}GB")

    step = 0
    last = ckpt.latest(out / "ckpt")
    if last is not None:
        meta = ckpt.load(last, pol, opt)
        step = int(meta["step"])
        log(rank, f"resumed from {last} (step {step})")
    start_step = step
    saver = ckpt.AsyncSaver(out / "ckpt", cfg.keep_ckpts, ck_group)

    pusher = WeightPusher(cfg.policy_urls, make_weight_source(pol), rank, cfg.wt_master_address,
                          cfg.wt_master_port, cfg.wt_buffer_bytes)
    info = pusher.push(f"step-{step}", dist.barrier)
    ctl0 = VLLMControl(cfg.policy_urls[0])
    if cfg.verify_sync_every and rank == 0:
        _, info["checksums_match_saved"] = _record_checksums(ctl0, out, step)
    log(rank, f"initial push step-{step}: {info}")

    roll = Rollouts(cfg, out) if rank == 0 else None
    wb = _init_wandb(cfg, out) if rank == 0 else None
    if roll is not None:
        log(rank, "question source:", json.dumps(roll.source.describe())[:500])
    metrics_f = open(out / "metrics.jsonl", "a") if rank == 0 else None

    rc = 0
    handles: dict[int, dict] = {}
    if rank == 0:
        for s in range(step + 1, min(step + cfg.rollout_ahead, cfg.total_steps) + 1):
            handles[s] = roll.submit(s, f"step-{step}", f"step-{step}")
    stopped = False
    val_due = val_running = None
    val_h = None
    try:
        while step < cfg.total_steps:
            nxt = step + 1
            t_roll = time.time()
            payload = [None] * world
            extra = {}
            if rank == 0:
                got = roll.wait(handles.pop(nxt))
                if got is None:
                    payload = [{"stop": True}] * world
                else:
                    samples, summaries, rinfo = got
                    items, fst = items_from_samples(samples, cfg.l_max)
                    if cfg.adv_mode == "raft":
                        items = [it for it in items if it.adv != 0.0]
                    mbs = split_minibatches(items, cfg.minibatches, cfg.seed * 1_000_003 + nxt)
                    per_mb = [balance(mb, world, cfg.pad_multiple) for mb in mbs]
                    n_tok = sum(it.n for it in items)
                    payload = [{"stop": False, "mbs": [sh[r] for sh in per_mb],
                                "n_tok": [sum(it.n for it in mb) for mb in mbs]} for r in range(world)]
                    advs = [it.adv for it in items]
                    scores = [m["score"] for m in summaries if m.get("score") is not None]
                    extra = {**rinfo, **fst, "n_items": len(items), "n_tok": n_tok,
                             "n_gens": sum(len(it.spans) if it.spans else 1 for it in items),
                             "tok_padded": sum(padded_len(len(it.ids), cfg.pad_multiple) for it in items),
                             "micro_batches": sum(len(sh[0]) for sh in per_mb), "minibatches": len(mbs),
                             "reward_mean": sum(scores) / len(scores) if scores else float("nan"),
                             "per_agent_mean": _mean([m.get("per_agent_mean") for m in summaries]),
                             "distinct_pairs": _mean([m.get("distinct_pairs") for m in summaries]),
                             "top_confidence_mean": _mean([m.get("top_confidence") for m in summaries]),
                             "vote_stake_mean": _mean([m.get("vote_stake") for m in summaries]),
                             "vote_free_mean": _mean([m.get("vote_free") for m in summaries]),
                             "confidence_gap": _mean([m.get("confidence_gap") for m in summaries]),
                             **({f"reward_{t}": _mean([m["score"] for m in summaries if m.get("task") == t])
                                 for t in {m.get("task") for m in summaries}} if cfg.mix else {}),
                             "exact_pairs": _mean([m.get("exact_pairs") for m in summaries]),
                             "precision": _mean([m.get("precision") for m in summaries]),
                             "recall": _mean([m.get("recall") for m in summaries]),
                             "adv_std": _std(advs), "rollout_s": rinfo["rollout_wall_s"],
                             "behav_lp": sum(sum(it.behavior) for it in items) / max(1, n_tok),
                             "rollout_wait_s": round(time.time() - t_roll, 1),
                             "staleness": step - int(rinfo["version"].rsplit("-", 1)[1])}
            mine = [None]
            dist.scatter_object_list(mine, payload if rank == 0 else None, src=0, group=gloo)
            mine = mine[0]
            if mine["stop"]:
                log(rank, "stop requested during rollouts; exiting without a step")
                stopped = True
                break
            if rank == 0 and cfg.async_rollout and nxt + cfg.rollout_ahead <= cfg.total_steps:
                ahead = nxt + cfg.rollout_ahead
                handles[ahead] = roll.submit(ahead, f"step-{step}", None)
            val_running, val_due = val_due, None
            if rank == 0 and val_running is not None:
                val_h = roll.submit_val(val_running, f"step-{val_running}")

            t_train = time.time()
            pol.train()
            acc = torch.zeros(5, device=dev, dtype=torch.float64)
            mbs = mine["mbs"]
            dummy = torch.zeros((1, cfg.pad_multiple), dtype=torch.long, device=dev)
            calls = 0

            def stop_due() -> bool:
                nonlocal calls
                calls += 1
                if calls % cfg.stop_check_every:
                    return False
                flag = torch.tensor([1.0 if STOP else 0.0], device=dev)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                return flag.item() > 0

            def logp(it):
                ids = torch.tensor([pad_right(it.ids, cfg.pad_multiple)], dtype=torch.long, device=dev)
                if cfg.ckpt_min_len and torch.is_grad_enabled():
                    set_checkpointing(pol, ids.shape[1] >= cfg.ckpt_min_len)
                return (pol(ids, torch.tensor(it.positions(), dtype=torch.long, device=dev)) if it.spans
                        else pol(ids, it.start, it.n))

            olds: list[list] = [[] for _ in mbs]
            probe, gnorms, applied, aborted = None, [], 0, False
            t_pre = time.time()
            if cfg.ppo_prepass:
                with torch.no_grad():
                    for j in range(1, len(mbs)):
                        for it in mbs[j]:
                            if stop_due():
                                aborted = True
                                break
                            if it is None:
                                pol(dummy, 1, 1)
                                olds[j].append(None)
                            else:
                                olds[j].append(logp(it).float())
                        if aborted:
                            break
            else:
                olds = [[None] * len(mb) for mb in mbs]
            prepass_s = time.time() - t_pre
            for j, mb in enumerate(mbs):
                if aborted:
                    break
                n_tok = max(1, mine["n_tok"][j])
                for i, it in enumerate(mb):
                    if stop_due():
                        aborted = True
                        break
                    if it is None:
                        (pol(dummy, 1, 1).sum() * 0.0).backward()
                        continue
                    lp = logp(it)
                    mu = torch.tensor(it.behavior, dtype=torch.float32, device=dev)
                    lt, st = token_pg(lp, mu, it.adv, cfg.eps_lo, cfg.eps_hi, cfg.tis_c, olds[j][i] if j else None)
                    loss = lt.sum() * (it.weight / n_tok)
                    loss.backward()
                    acc += torch.stack([st["k3_sum"].double(), st["tis_clipped"].double(),
                                        torch.tensor(float(it.n), device=dev, dtype=torch.float64),
                                        loss.detach().double(), st["ppo_clipped"].double()])
                    probe = it
                if aborted:
                    break
                gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                gnorms.append(float(gnorm.full_tensor() if hasattr(gnorm, "full_tensor") else gnorm))
                opt.step()
                opt.zero_grad(set_to_none=True)
                applied += 1
            if aborted:
                opt.zero_grad(set_to_none=True)
                stopped = True
                if applied == 0:
                    log(rank, f"stop requested inside update {nxt}; abandoning it")
                    break
                step = nxt
                log(rank, f"stop requested inside update {nxt}; keeping it truncated after {applied}/{len(mbs)} "
                          f"mini-batches")
                break
            gnorm = sum(gnorms) / max(1, len(gnorms))
            dist.all_reduce(acc)
            step = nxt
            train_s = time.time() - t_train

            val = None
            drain = cfg.async_rollout and cfg.rollout_ahead == 1 and step < cfg.total_steps
            if drain or val_running is not None:
                if rank == 0:
                    t_w = time.time()
                    if drain:
                        roll.wait(handles[step + 1])
                    if val_h is not None:
                        val = roll.val_metrics(val_h)
                        if val is not None:
                            val["val_step"] = val_running
                        val_h = None
                    extra["overlap_wait_s"] = round(time.time() - t_w, 1)
                dist.barrier(group=gloo)
            check = cfg.verify_sync_every and step % cfg.verify_sync_every == 0
            before = ctl0.checksums() if (check and rank == 0) else None
            sync = pusher.push(f"step-{step}", dist.barrier)
            if check:
                after, match = _record_checksums(ctl0, out, step) if rank == 0 else (None, None)
                if match is not None:
                    sync["checksums_match_saved"] = match
                sync.update(_verify(pol, probe if rank == 0 else None, ctl0, cfg, before, after, rank, dev))
            saver.poll()
            if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
                sync["ckpt_s"] = round(saver.save(step, pol, opt, {"version": f"step-{step}"}), 1)
                if step == cfg.total_steps:
                    saver.wait()
            if saver.last_write_s is not None:
                sync["ckpt_write_s"], saver.last_write_s = round(saver.last_write_s, 1), None
            if cfg.hf_every and (step % cfg.hf_every == 0 or step == cfg.total_steps):
                th = time.time()
                _export_milestone(pol, cfg, out, step, rank)
                sync["hf_s"] = round(time.time() - th, 1)
            if cfg.val_every and (step % cfg.val_every == 0 or step == cfg.total_steps):
                if cfg.async_rollout and step < cfg.total_steps:
                    val_due = step
                elif rank == 0:
                    v = roll.validate(step, f"step-{step}")
                    if v is not None:
                        val = {**v, "val_step": step}
            if rank == 0:
                a = acc.tolist()
                core = {"reward_mean": extra["reward_mean"], "adv_std": extra["adv_std"], "gnorm": float(gnorm),
                        "per_agent_mean": extra["per_agent_mean"], "distinct_pairs": extra["distinct_pairs"],
                        "behav_lp": extra["behav_lp"], "mismatch_k3": a[0] / max(1.0, a[2]), "tis_clipped_frac": a[1] / max(1.0, a[2]),
                        "ppo_clip_frac": a[4] / max(1.0, a[2]),
                        "dropped_len_frac": extra["dropped_len"] / max(1, extra["n_samples"]), **(val or {})}
                rec = {"step": step, **core, "loss": a[3] / max(1, len(mbs)), **extra, "train_s": round(train_s, 1),
                       "prepass_s": round(prepass_s, 1), **sync}
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
                log(rank, f"step {step} " + json.dumps(rec))
                if wb is not None:
                    wb.log({k: v for k, v in core.items() if v is not None}, step=step)
            flag = torch.tensor([1.0 if STOP else 0.0], device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item() > 0:
                log(rank, f"stop requested; exiting after step {step}")
                stopped = True
                break
            if rank == 0 and not cfg.async_rollout and step < cfg.total_steps:
                handles[step + 1] = roll.submit(step + 1, f"step-{step}", None)
        if stopped:
            saver.wait()
            dist.barrier(group=gloo)
            last = ckpt.latest(out / "ckpt")
            if step > 0 and (last is None or int(last.name.split("_")[1]) < step):
                tc = time.time()
                ckpt.save(out / "ckpt", step, pol, opt, {"version": f"step-{step}"}, cfg.keep_ckpts)
                log(rank, f"checkpoint of step {step} written on stop in {time.time() - tc:.0f}s")
    except Exception:
        rc = 1
        raise
    finally:
        pusher.shutdown()
        if roll is not None:
            roll.close()
        if wb is not None:
            wb.finish()
        if metrics_f is not None:
            metrics_f.close()
        if rc == 0:
            dist.barrier()
        dist.destroy_process_group()
    return rc


def _export_milestone(pol, cfg, out: Path, step: int, rank: int) -> None:
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    from rci.train.hf_export import write_hf

    sd = get_model_state_dict(pol, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        write_hf(sd, Path(cfg.model_path), out / "hf" / f"step_{step:06d}",
                 {"run": str(out), "step": step, "run_id": cfg.run_id, "source": "trainer"})
    del sd
    dist.barrier()


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _std(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _record_checksums(ctl, out: Path, step: int):
    cs = ctl.checksums()
    p = out / "checksums" / f"step_{step:06d}.json"
    if p.exists():
        return cs, json.loads(p.read_text()) == cs
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cs))
    return cs, None


def _verify(pol, probe, ctl, cfg, before: dict | None, after: dict | None, rank: int, dev) -> dict:
    import torch

    from rci.train.loss import pad_right

    with torch.no_grad():
        if rank == 0 and probe is not None:
            ids = torch.tensor([pad_right(probe.ids, cfg.pad_multiple)], dtype=torch.long, device=dev)
            lp = pol(ids, probe.start, len(probe.ids) - probe.start).double().cpu()
        else:
            pol(torch.zeros((1, cfg.pad_multiple), dtype=torch.long, device=dev), 1, 1)
            lp = None
    if rank != 0 or probe is None:
        return {}
    v = torch.tensor(ctl.prompt_logprobs(cfg.served_policy, probe.ids, probe.start), dtype=torch.float64)
    d = (v - lp).abs()
    out = {"sync_check_mean_abs": round(d.mean().item(), 5), "sync_check_max_abs": round(d.max().item(), 4),
           "sync_check_n": int(d.numel())}
    if before and after:
        out["weights_changed"] = sum(1 for k in after if before.get(k) != after[k])
        out["weights_total"] = len(after)
    return out


def _init_wandb(cfg, out: Path):
    if cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb

        return wandb.init(project=cfg.wandb_project, name=cfg.run_id, id=cfg.run_id.replace("/", "-"),
                          resume="allow", mode=cfg.wandb_mode, dir=str(out),
                          config={k: v for k, v in dataclasses.asdict(cfg).items()
                                  if k not in ("policy_urls", "peer_urls")})
    except Exception as e:
        print(f"[wandb] disabled: {e!r}", flush=True)
        return None


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
