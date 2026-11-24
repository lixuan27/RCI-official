"""Candidates by weight interpolation: theta(alpha) = theta_g + alpha (theta_trained - theta_g).

alpha = 1 is the training's own export; alpha = 1/2 steps half way, alpha = 2 extrapolates. Every floating tensor is
computed in float64 and stored once in the output dtype (bfloat16 rounded to nearest even); an element the training
left equal to theta_g is kept bit-identical. Non-floating tensors must equal theta_g's. Reads and writes safetensors
directly with numpy: no torch is needed.

    python -m rci.chains.merge --trained DIR --base DIR --out DIR --alpha 1/2 [--dtype keep|BF16|F32]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

FLOATS = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": None}
SIZES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
         "U16": 2, "U32": 4, "U64": 8}
CHUNK = 1 << 22


class MergeError(Exception):
    pass


def read_header(path) -> tuple:
    """(tensor table without __metadata__, the offset where the data starts)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise MergeError(f"{path}: not a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        hb = f.read(n)
    if len(hb) != n:
        raise MergeError(f"{path}: truncated header")
    hd = json.loads(hb.decode("utf-8"))
    hd.pop("__metadata__", None)
    return hd, 8 + n


def write_safetensors(path, tensors: list) -> None:
    """tensors: [(name, dtype, shape, bytes)] in order."""
    hd, pos = {}, 0
    for name, dt, shape, data in tensors:
        hd[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [pos, pos + len(data)]}
        pos += len(data)
    hd["__metadata__"] = {"format": "pt"}
    hb = json.dumps(hd, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, _, _, data in tensors:
            f.write(data)


def bf16_decode(buf) -> np.ndarray:
    return (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def bf16_encode(x32: np.ndarray) -> bytes:
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()


def decode(buf, dt: str) -> np.ndarray:
    if dt == "BF16":
        return bf16_decode(buf).astype(np.float64)
    return np.frombuffer(buf, dtype=FLOATS[dt]).astype(np.float64)


def encode(x64: np.ndarray, dt: str) -> bytes:
    if dt == "BF16":
        return bf16_encode(x64.astype(np.float32))
    return x64.astype(FLOATS[dt]).tobytes()


def blend(t64: np.ndarray, b64: np.ndarray, alpha: Fraction) -> np.ndarray:
    return np.where(t64 == b64, t64, b64 + (t64 - b64) * alpha.numerator / alpha.denominator)


def index_of(model_dir) -> dict:
    """{tensor name: (file, dtype, shape, start, end)} over a directory's shards."""
    out = {}
    for p in sorted(Path(model_dir).glob("*.safetensors")):
        hd, off = read_header(p)
        for name, t in hd.items():
            if name in out:
                raise MergeError(f"{model_dir}: tensor {name} in two shards")
            out[name] = (str(p), t["dtype"], list(t["shape"]), off + t["data_offsets"][0], off + t["data_offsets"][1])
    if not out:
        raise MergeError(f"{model_dir}: no safetensors shard")
    return out


def _read(path: str, a: int, b: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(a)
        return f.read(b - a)


def merge_file(trained: Path, base: dict, out: Path, alpha: Fraction, dtype: str) -> int:
    hd, off = read_header(trained)
    tensors, total = [], 0
    for name, t in sorted(hd.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        dt = t["dtype"]
        if dt not in SIZES:
            raise MergeError(f"{trained}: {name} has dtype {dt}")
        if name not in base:
            raise MergeError(f"{trained}: tensor {name} is not in the base")
        bp, bdt, bshape, ba, bb = base[name]
        if list(t["shape"]) != bshape:
            raise MergeError(f"{name}: shape {t['shape']} here, {bshape} in the base")
        tb = _read(str(trained), off + t["data_offsets"][0], off + t["data_offsets"][1])
        bbuf = _read(bp, ba, bb)
        if dt not in FLOATS or bdt not in FLOATS:
            if dt != bdt or tb != bbuf:
                raise MergeError(f"{name}: a non-floating tensor that differs from the base's")
            tensors.append((name, dt, t["shape"], tb))
            continue
        odt = dt if dtype == "keep" else dtype
        n = len(tb) // SIZES[dt]
        if len(bbuf) // SIZES[bdt] != n:
            raise MergeError(f"{name}: element counts differ")
        parts = []
        for i in range(0, n, CHUNK):
            k = min(CHUNK, n - i)
            t64 = decode(tb[i * SIZES[dt]:(i + k) * SIZES[dt]], dt)
            b64 = decode(bbuf[i * SIZES[bdt]:(i + k) * SIZES[bdt]], bdt)
            parts.append(encode(t64 if alpha == 1 else blend(t64, b64, alpha), odt))
        data = b"".join(parts)
        tensors.append((name, odt, t["shape"], data))
        total += len(data)
    write_safetensors(out, tensors)
    return total


def merge(trained, base, out, alpha, dtype: str = "keep") -> dict:
    """Write the candidate at `alpha` to `out` (a temporary name, then renamed; an existing `out` is kept)."""
    alpha = Fraction(str(alpha))
    trained, out = Path(trained), Path(out)
    if out.exists():
        return json.loads((out / "ALPHA.json").read_text()) if (out / "ALPHA.json").exists() else {"kept": str(out)}
    if dtype not in ("keep",) + tuple(FLOATS):
        raise MergeError(f"dtype {dtype!r}")
    bidx = index_of(base)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    total = 0
    shards = sorted(trained.glob("*.safetensors"))
    if not shards:
        raise MergeError(f"{trained}: no safetensors shard")
    for p in shards:
        total += merge_file(p, bidx, tmp / p.name, alpha, dtype)
    for f in trained.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and not (tmp / f.name).exists():
            shutil.copy2(f, tmp / f.name)
    idx = tmp / "model.safetensors.index.json"
    if idx.exists():
        d = json.loads(idx.read_text())
        d.setdefault("metadata", {})["total_size"] = total
        idx.write_text(json.dumps(d, indent=2) + "\n")
    meta = {"alpha": str(alpha), "trained": str(trained), "base": str(base), "dtype": dtype}
    (tmp / "ALPHA.json").write_text(json.dumps(meta, indent=1) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, out)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rci.chains.merge", description=__doc__.splitlines()[0])
    ap.add_argument("--trained", required=True)
    ap.add_argument("--base", required=True, help="the current generation's weights theta_g")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", required=True)
    ap.add_argument("--dtype", default="keep", choices=("keep",) + tuple(FLOATS))
    a = ap.parse_args(argv)
    print(json.dumps(merge(a.trained, a.base, a.out, a.alpha, a.dtype)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
