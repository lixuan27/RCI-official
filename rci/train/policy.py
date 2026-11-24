"""FSDP2 policy model: HF Qwen3.5 (vision tower frozen, text-only input) with chunked lm-head log-probs."""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


class PolicyLogProb(torch.nn.Module):
    def __init__(self, hf, chunk: int = 2048):
        super().__init__()
        self.hf, self.chunk = hf, chunk

    @staticmethod
    def _lp(h, t, w):
        return torch.nn.functional.linear(h, w).float().log_softmax(-1).gather(-1, t[:, None])[:, 0]

    def forward(self, ids: torch.Tensor, start, n: int | None = None) -> torch.Tensor:
        hid = self.hf.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        if n is None:
            h, t = hid[start - 1], ids[0, start]
            n = int(start.numel())
        else:
            h, t = hid[start - 1:start - 1 + n], ids[0, start:start + n]
        w = self.hf.lm_head.weight
        return torch.cat([checkpoint(self._lp, h[s:s + self.chunk], t[s:s + self.chunk], w, use_reentrant=False)
                          for s in range(0, n, self.chunk)])


def set_checkpointing(pol, on: bool) -> None:
    """Turn activation checkpointing of the decoder layers on or off for the next forward/backward."""
    layers = getattr(pol, "_ckpt_layers", None)
    if layers is None:
        layers = [m for m in pol.modules() if hasattr(m, "_gradient_checkpointing_func")]
        pol._ckpt_layers = layers
    for m in layers:
        m.gradient_checkpointing = on


def _text_layers(hf):
    m = hf.model
    lm = getattr(m, "language_model", m)
    return lm.layers


def build_policy(model_path: str, mesh, chunk: int = 2048, master_dtype=torch.float32):
    """Load on CPU in the master dtype, freeze the vision tower, wrap with FSDP2 (moves shards to the GPU)."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path)
    if getattr(cfg, "model_type", "") == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration as Cls
    else:
        from transformers import AutoModelForCausalLM as Cls
    hf = Cls.from_pretrained(model_path, dtype=master_dtype, attn_implementation="sdpa", device_map="cpu")
    hf.config.use_cache = False
    visual = getattr(hf.model, "visual", None)
    if visual is not None:
        visual.requires_grad_(False)
    hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in _text_layers(hf):
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    if visual is not None:
        for blk in getattr(visual, "blocks", []):
            fully_shard(blk, mesh=mesh, mp_policy=mp)
    pol = PolicyLogProb(hf, chunk=chunk)
    fully_shard(pol, mesh=mesh, mp_policy=mp)
    dev = torch.device("cuda", torch.cuda.current_device())
    for mod in pol.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.device.type != "cuda":
                mod._buffers[name] = buf.to(dev)
    bad = [n for n, p in pol.named_parameters() if p.device.type != "cuda"]
    if bad:
        raise RuntimeError(f"parameters left off the GPU after fully_shard: {bad[:5]}")
    return pol


def make_weight_source(pol):
    from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource, materialize_full_tensor

    class Bf16TextSource(WeightSource):
        def _named(self):
            for n, p in pol.named_parameters():
                n = n.removeprefix("hf.")
                if not n.startswith("model.visual."):
                    yield n, p

        def metadata(self):
            return [ParamMeta(n, torch.bfloat16, tuple(p.shape)) for n, p in self._named()]

        def __iter__(self):
            with torch.no_grad():
                for n, p in self._named():
                    yield n, materialize_full_tensor(p.detach()).to(torch.bfloat16)

    return Bf16TextSource()
