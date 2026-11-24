"""Model families: vLLM serve flags, the thinking switch and recommended sampling per served model."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass, field

from rci import paths

EFFORTS = ("low", "medium", "high")
GPU_MEMORY_UTILIZATION = 0.90
MIN_KV_GIB = 10.0
REFUSED = 3


class FamilyError(ValueError):
    pass


@dataclass
class Family:
    name: str
    reasoning_parser: str
    tool_parser: str
    thinking: str = "template"
    thinking_key: str = "enable_thinking"
    effort_on: str | None = None
    effort_off: str | None = None
    task_effort: dict = field(default_factory=dict)
    no_think_min_tokens: int | None = None
    sampling: dict = field(default_factory=dict)
    sampling_source: str = ""
    sampling_verified: bool = False
    model_types: tuple = ()
    tp: dict = field(default_factory=dict)
    weight_gb: float | None = None
    max_model_len: int | None = None
    serve_extra: tuple = ()
    env: dict = field(default_factory=dict)
    encodings: tuple = ()
    named_tool_choice: bool = True
    reference: str = ""

    @property
    def parser_flags(self) -> list[str]:
        return ["--reasoning-parser", self.reasoning_parser, "--enable-auto-tool-choice", "--tool-call-parser",
                self.tool_parser]

    def effort(self, enable_thinking: bool | None, requested: str | None = None) -> str | None:
        if self.thinking != "effort":
            return None
        if enable_thinking is False:
            return self.effort_off
        if enable_thinking is None:
            return requested
        return requested or self.effort_on

    def tp_for(self, gpu: str | None) -> int | None:
        g = (gpu or "").upper()
        for kind in sorted(self.tp, key=len, reverse=True):
            if kind.upper() in g:
                return self.tp[kind]
        return None

    def resolve_tp(self, gpu: str | None, mem_mib: int | None = None, weight_bytes: float | None = None) -> int:
        tp = self.tp_for(gpu)
        if tp is None:
            raise FamilyError(f"family {self.name}: GPU kind {gpu!r} is not registered (known: {sorted(self.tp)}); "
                              "set TP explicitly")
        wb = weight_bytes if weight_bytes else (self.weight_gb or 0) * 1e9
        if mem_mib and wb:
            kv = GPU_MEMORY_UTILIZATION * mem_mib / 1024 - wb / 2 ** 30 / tp
            if kv < MIN_KV_GIB:
                raise FamilyError(f"family {self.name}: TP {tp} on {gpu!r} ({mem_mib} MiB) leaves {kv:.1f} GiB of KV "
                                  f"cache per card (< {MIN_KV_GIB}); set TP explicitly")
        return tp


O200K = ("o200k_base.tiktoken", "fb374d419588a4632f3f557e76b4b70aebbca790")

FAMILIES: dict[str, Family] = {
    "qwen": Family(
        name="qwen", reasoning_parser="qwen3", tool_parser="qwen3_coder",
        sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        sampling_source="Qwen3 generation_config.json (top-p 0.95, top-k 20) at temperature 1.0",
        sampling_verified=True, model_types=("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe", "qwen3_next"),
        reference="Qwen model card, thinking mode"),
    "gptoss": Family(
        name="gptoss", reasoning_parser="openai_gptoss", tool_parser="openai", thinking="effort",
        effort_on="medium", effort_off="low", task_effort={"gpqa": "high"}, no_think_min_tokens=4096,
        sampling={"temperature": 1.0, "top_p": 1.0},
        sampling_source="gpt-oss model card: temperature 1.0, top-p 1.0",
        sampling_verified=True, model_types=("gpt_oss",),
        tp={"H200": 1, "H100": 1, "A800": 2, "A100": 2}, weight_gb=65.25, max_model_len=131072,
        encodings=(O200K,), reference="gpt-oss model card"),
    "glm45": Family(
        name="glm45", reasoning_parser="glm45", tool_parser="glm45",
        sampling={"temperature": 0.6, "top_p": 0.95},
        sampling_source="GLM-4.5 served at temperature 0.6, top-p 0.95",
        sampling_verified=False, model_types=("glm4_moe",),
        tp={"H200": 2, "H100": 4, "A800": 4, "A100": 4}, weight_gb=112.56, max_model_len=131072,
        env={"VLLM_ENFORCE_STRICT_TOOL_CALLING": "0"}, named_tool_choice=False,
        reference="GLM-4.5 model card"),
}

MODELS: dict[str, tuple[str, str]] = {
    "qwen3.5-4b": ("qwen", "Qwen3.5-4B"),
    "qwen3.5-9b": ("qwen", "Qwen3.5-9B"),
    "qwen3.8-27b": ("qwen", "Qwen3.8-27B"),
    "qwen3.6-35b-a3b": ("qwen", "Qwen3.6-35B-A3B"),
    "gpt-oss-120b": ("gptoss", "gpt-oss-120b"),
    "glm-4.5-air-fp8": ("glm45", "GLM-4.5-Air-FP8"),
}
PREFIXES = (("gpt-oss", "gptoss"), ("glm-4.5", "glm45"), ("glm-4.6", "glm45"), ("glm-4.7", "glm45"))


def family_name(model: str | None) -> str:
    m = str(model or "").strip().lower()
    if m in MODELS:
        return MODELS[m][0]
    for p, fam in PREFIXES:
        if m.startswith(p):
            return fam
    return "qwen"


def family_of(model: str | None) -> Family:
    return FAMILIES[family_name(model)]


def model_path(model: str) -> str:
    """Weights of a served name: a base model under RCI_MODELS, else a trained checkpoint's HF export under RCI_CKPT."""
    m = str(model).strip()
    if m.lower() in MODELS:
        return str(paths.models_dir(MODELS[m.lower()][1]))
    hf = paths.ckpt_dir(m, "hf")
    if hf.is_dir():
        return str(hf)
    return str(paths.models_dir(m))


def serve_env(f: Family) -> dict:
    """Environment variables exported for `vllm serve` of a family."""
    out = dict(f.env)
    date = paths.env("RCI_GPTOSS_SYSTEM_DATE")
    if f.name == "gptoss" and date:
        out["VLLM_SYSTEM_START_DATE"] = date
    return out


def record_info(model: str | None, reasoning_effort: str | None = None, thinking: bool = True) -> dict | None:
    """What a record stores about a non-Qwen family (None for Qwen): the family and the efforts its calls ran at."""
    f = family_of(model)
    if f.name == "qwen":
        return None
    out = {"name": f.name}
    if f.thinking == "effort":
        out["reasoning_effort"] = (reasoning_effort or f.effort_on) if thinking else f.effort_off
        out["effort_off"] = f.effort_off
    return out


def index_weight_bytes(path: str | None) -> float | None:
    if not path:
        return None
    try:
        with open(os.path.join(path, "model.safetensors.index.json")) as fh:
            v = float(json.load(fh)["metadata"]["total_size"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return v if v > 0 else None


def shell(model: str, gpu: str | None = None, environ=None, gpu_mem: int | None = None,
          path: str | None = None) -> str:
    """Shell assignments for a serve script; FamilyError when the tensor-parallel size cannot be resolved."""
    environ = os.environ if environ is None else environ
    f = family_of(model)
    q = shlex.quote
    path = path or model_path(model)
    out = [f"FAMILY={f.name}", f"MODEL_PATH={q(path)}"]
    out.append(f"FAMILY_FLAGS={q(environ.get('FAMILY_FLAGS') or ' '.join(f.parser_flags))}")
    if environ.get("TP"):
        out.append(f"FAMILY_TP={q(environ['TP'])}")
    elif f.tp:
        out.append(f"FAMILY_TP={f.resolve_tp(gpu, gpu_mem, index_weight_bytes(path))}")
    else:
        out.append("FAMILY_TP=")
    out.append(f"FAMILY_MAX_MODEL_LEN={f.max_model_len or ''}")
    out.append(f"FAMILY_SERVE_EXTRA={q(' '.join(f.serve_extra))}")
    out.append(f"FAMILY_ENCODINGS={q(' '.join(f'{name}:{key}' for name, key in f.encodings))}")
    env = serve_env(f)
    out.append(f"FAMILY_ENV={q(' '.join(f'{k}={v}' for k, v in sorted(env.items())))}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rci.agents.families",
                                 description="Print the serving variables of a model family.")
    ap.add_argument("what", choices=["shell", "flags", "info"])
    ap.add_argument("model", help="served model name")
    ap.add_argument("--gpu", default=None, help="the GPU name nvidia-smi reports (picks the tensor-parallel size)")
    ap.add_argument("--gpu-mem", default=None, help="the card's memory in MiB (checks the size fits)")
    ap.add_argument("--model-path", default=None, help="weights directory (default: resolved from the served name)")
    a = ap.parse_args(argv)
    if a.what == "shell":
        mem = (a.gpu_mem or "").strip()
        try:
            s = shell(a.model, a.gpu, gpu_mem=int(mem) if mem.isdigit() else None, path=a.model_path)
        except FamilyError as e:
            print(f"[families] {e}", file=sys.stderr)
            return REFUSED
        print(s)
    elif a.what == "flags":
        print(" ".join(family_of(a.model).parser_flags))
    else:
        print(json.dumps(asdict(family_of(a.model)), indent=1))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
