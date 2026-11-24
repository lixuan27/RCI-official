"""Weight sync from the trainer to vLLM servers through their native NCCL weight-transfer API."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class VLLMControl:
    def __init__(self, base_url: str, timeout: float = 1800.0):
        self.base = base_url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.timeout = timeout

    def _req(self, method: str, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with _OPENER.open(req, timeout=self.timeout) as r:
            raw = r.read()
        return json.loads(raw) if raw else {}

    def post(self, path: str, body: dict | None = None):
        return self._req("POST", path, body)

    def get(self, path: str):
        return self._req("GET", path)

    def world_size(self) -> int:
        return int(self.get("/get_world_size?include_dp=true")["world_size"])

    def pause(self, mode: str = "wait"):
        return self.post(f"/pause?mode={mode}")

    def resume(self):
        return self.post("/resume")

    def set_version(self, v: str):
        return self.post("/update_weight_version", {"new_version": v})

    def version(self) -> str | None:
        return self.get("/weight_info").get("weight_version")

    def reset_prefix_cache(self) -> bool:
        return bool(self.post("/reset_prefix_cache").get("success", False))

    def checksums(self) -> dict:
        return self.post("/weight_checker", {"action": "checksum"}).get("checksums", {})

    def prompt_logprobs(self, model: str, ids: list[int], start: int) -> list[float]:
        """Log-probs of ids[start:] given their prefixes (prefill path, no sampling)."""
        out = self.post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
                                             "prompt_logprobs": 1})
        pl = out["choices"][0]["prompt_logprobs"]
        return [pl[j][str(ids[j])]["logprob"] for j in range(start, len(ids))]


class WeightPusher:
    def __init__(self, urls: list[str], source, rank: int, master_address: str, master_port: int,
                 buffer_bytes: int):
        from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
        from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

        self.rank = rank
        self.ctls = [VLLMControl(u) for u in urls]
        self.engines = []
        for i, (u, ctl) in enumerate(zip(urls, self.ctls)):
            n = 0
            if rank == 0:
                try:
                    ctl.post("/shutdown_weight_transfer_engine")
                except Exception:
                    pass
                n = ctl.world_size()
            info = NCCLTrainerInitInfo(rank=rank, master_address=master_address, master_port=master_port + i,
                                       world_size=1 + n, packed=True, packed_buffer_size_bytes=buffer_bytes)
            self.engines.append(WeightTransferTrainerFactory.trainer_init(
                info, client=HTTPVLLMWeightSyncClient(ctl.base, timeout=1800), source=source))

    def push(self, version: str, barrier) -> dict:
        import time

        t0 = time.time()
        if self.rank == 0:
            for ctl in self.ctls:
                ctl.pause("wait")
        barrier()
        for eng in self.engines:
            eng.send_weights()
        reset_ok = True
        if self.rank == 0:
            for ctl in self.ctls:
                ctl.set_version(version)
                reset_ok &= ctl.reset_prefix_cache()
                ctl.resume()
        barrier()
        return {"sync_s": round(time.time() - t0, 2), "reset_ok": reset_ok}

    def shutdown(self) -> None:
        for eng in self.engines:
            try:
                eng.shutdown()
            except Exception:
                pass
        if self.rank == 0:
            for ctl in self.ctls:
                try:
                    ctl.post("/shutdown_weight_transfer_engine")
                except Exception:
                    pass
