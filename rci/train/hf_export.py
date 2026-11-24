"""Export policy weights to a Hugging Face directory that vLLM can serve."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

WEIGHT_FILES = ("model.safetensors.index.json", "config.json")


def hf_class(base: Path):
    from transformers import AutoConfig

    if getattr(AutoConfig.from_pretrained(base), "model_type", "") == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration as Cls
    else:
        from transformers import AutoModelForCausalLM as Cls
    return Cls


def write_hf(state: dict, base: Path, out: Path, meta: dict, dtype: str = "bfloat16") -> Path:
    """state: full (unsharded) policy state dict with PolicyLogProb keys ("hf." prefix) or plain HF keys.
    dtype "float32" writes a full-precision master that a later training continues from."""
    import torch

    base, out = Path(base), Path(out)
    dt = getattr(torch, dtype)
    hf = hf_class(base).from_pretrained(base, dtype=dt, device_map="cpu")
    sd = {k.removeprefix("hf."): v for k, v in state.items()}
    missing, unexpected = hf.load_state_dict({k: v.to(dt) for k, v in sd.items()}, strict=False)
    tk = getattr(hf, "_tied_weights_keys", None) or []
    tied = set(tk.keys() if isinstance(tk, dict) else tk)
    missing = [k for k in missing if k not in tied]
    if missing or unexpected:
        raise RuntimeError(f"state dict does not match {base.name}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    hf.save_pretrained(tmp, safe_serialization=True, max_shard_size="5GB")
    for f in base.iterdir():
        if f.is_file() and f.suffix in (".json", ".jinja", ".txt", ".model") and f.name not in WEIGHT_FILES \
                and not (tmp / f.name).exists():
            shutil.copy2(f, tmp / f.name)
    (tmp / "EXPORT.json").write_text(json.dumps({**meta, "base": base.name}, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export a training checkpoint of rci.train.loop to <run>/hf/step_<n>.")
    ap.add_argument("--run", required=True, help="the run directory (holds config.json and ckpt/)")
    ap.add_argument("--step", type=int, default=None, help="default: the latest complete checkpoint")
    a = ap.parse_args(argv)
    import torch
    import torch.distributed.checkpoint as dcp

    from rci.train import ckpt as ck
    from rci.train.policy import PolicyLogProb

    run = Path(a.run).resolve()
    cfg = json.loads((run / "config.json").read_text())
    src = (run / "ckpt" / f"step_{a.step:06d}") if a.step is not None else ck.latest(run / "ckpt")
    if src is None or not (src / "meta.json").exists():
        raise SystemExit(f"no complete checkpoint under {run / 'ckpt'}")
    step = int(json.loads((src / "meta.json").read_text())["step"])
    out = run / "hf" / f"step_{step:06d}"
    base = Path(cfg["model_path"])
    hf = hf_class(base).from_pretrained(base, dtype=torch.float32, device_map="cpu")
    pol = PolicyLogProb(hf)
    sd = pol.state_dict()
    state = {"model": sd}
    dcp.load(state, checkpoint_id=str(src))
    missing = [k for k in sd if k not in state["model"]]
    if missing:
        raise SystemExit(f"keys missing from the checkpoint: {missing[:5]}")
    write_hf(state["model"], base, out, {"step": step, "run_id": cfg.get("run_id")})
    print(f"exported step {step} -> {out}")
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
