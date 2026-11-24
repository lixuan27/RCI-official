"""Run a SWE-bench image pulled as OCI layers without docker, root, setuid helpers or FUSE."""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tarfile
import threading
import time


def rmtree_ns(path: str) -> None:
    if os.path.lexists(path):
        subprocess.run(["unshare", "-Ur", "rm", "-rf", "--", path], check=True)


def _filter(m: tarfile.TarInfo, dest: str):
    m = tarfile.tar_filter(m, dest)
    if m.isdir():
        return m.replace(mode=(m.mode or 0) | 0o700, deep=False)
    if m.isreg():
        return m.replace(mode=(m.mode or 0) | 0o600, deep=False)
    return m


_LAYER_LOCKS: dict = {}
_LAYER_LOCKS_LOCK = threading.Lock()


def extract_layer(blob: str, dest: str) -> dict | None:
    """Extract one layer tar(.gz) into dest (atomic via rename)."""
    if os.path.isdir(dest):
        return None
    with _LAYER_LOCKS_LOCK:
        lock = _LAYER_LOCKS.setdefault(dest, threading.Lock())
    with lock:
        if os.path.isdir(dest):
            return None
        return _extract_layer(blob, dest)


def _extract_layer(blob: str, dest: str) -> dict | None:
    tmp = f"{dest}.tmp{os.getpid()}.{threading.get_ident()}"
    rmtree_ns(tmp)
    os.makedirs(tmp)
    real_tmp = os.path.realpath(tmp)
    st = {"members": 0, "whiteouts": 0, "opaque": 0, "skipped_dev": 0, "refused": 0}
    t0 = time.time()
    with tarfile.open(blob, "r|*") as tf:
        for m in tf:
            name = m.name.lstrip("/")
            while name.startswith("./"):
                name = name[2:]
            if not name or name == ".":
                continue
            d, b = os.path.split(name)
            if b.startswith(".wh."):
                pd = os.path.join(tmp, d)
                if ".." in name.split("/") or not os.path.realpath(pd).startswith(real_tmp):
                    st["refused"] += 1
                    continue
                os.makedirs(pd, exist_ok=True)
                if b == ".wh..wh..opq":
                    os.setxattr(pd, "user.overlay.opaque", b"y")
                    st["opaque"] += 1
                else:
                    os.mknod(os.path.join(pd, b[4:]), stat.S_IFCHR | 0o600, os.makedev(0, 0))
                    st["whiteouts"] += 1
                continue
            if m.ischr() or m.isblk():
                st["skipped_dev"] += 1
                continue
            try:
                tf.extract(m, tmp, filter=_filter)
                st["members"] += 1
            except tarfile.FilterError as e:
                st["refused"] += 1
                print(f"refused {name}: {e}", file=sys.stderr)
    try:
        os.rename(tmp, dest)
    except OSError:
        if os.path.isdir(dest):
            rmtree_ns(tmp)
            return None
        raise
    st["secs"] = round(time.time() - t0, 1)
    return st


def image_record(store: str, key: str) -> dict:
    with open(os.path.join(store, "images", key + ".json")) as f:
        return json.load(f)


def prepare_layers(store: str, key: str, cache: str, log=print) -> str:
    """Extract all layers of image `key`; return overlay lowerdir (top first)."""
    rec = image_record(store, key)
    dirs = []
    for desc in rec["manifest"]["layers"]:
        hexd = desc["digest"].split(":", 1)[1]
        dest = os.path.join(cache, "layers", hexd)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        st = extract_layer(os.path.join(store, "blobs", "sha256", hexd), dest)
        if st is not None:
            log(f"layer {hexd[:12]} {desc['size']/1e6:.0f} MB -> {json.dumps(st)}")
        dirs.append(dest)
    return ":".join(reversed(dirs))


_SH = r'''
set -euo pipefail
R="$SWE_RUN/root"
mkdir -p "$SWE_RUN/upper" "$SWE_RUN/work" "$R"
opts="lowerdir=$SWE_LOWER,upperdir=$SWE_RUN/upper,workdir=$SWE_RUN/work"
mount -t overlay overlay -o "userxattr,$opts" "$R"
for d in dev proc sys swe_io etc tmp; do mkdir -p "$R/$d"; done
mount --rbind /dev "$R/dev"
mount -t proc proc "$R/proc"
mount --rbind /sys "$R/sys"
mount --bind "$SWE_IO" "$R/swe_io"
rm -f "$R/etc/resolv.conf"; cat /etc/resolv.conf > "$R/etc/resolv.conf"
exec chroot "$R" "$@"
'''


def run(store: str, key: str, cache: str, rundir: str, io: str, cmd: list[str],
        timeout: float | None = None, keep: bool = False, extra_env: dict | None = None) -> int:
    rec = image_record(store, key)
    cfg = rec["config"].get("config", {})
    env = list(cfg.get("Env") or ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"])
    env += ["HOME=/root", "HOSTNAME=swe-sandbox"] + [f"{k}={v}" for k, v in (extra_env or {}).items()]
    wd = cfg.get("WorkingDir") or "/"
    lower = prepare_layers(store, key, cache)
    rmtree_ns(rundir)
    os.makedirs(rundir)
    os.makedirs(io, exist_ok=True)
    inner = ["/usr/bin/env", "-i", *env, "/bin/sh", "-c", 'cd "$0" && exec "$@"', wd, *cmd]
    penv = dict(os.environ, SWE_RUN=rundir, SWE_LOWER=lower, SWE_IO=os.path.abspath(io))
    argv = ["unshare", "-Urmpf", "--kill-child", "--propagation", "private",
            "bash", "-c", _SH, "bash", *inner]
    try:
        return subprocess.run(argv, env=penv, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124
    finally:
        if not keep:
            rmtree_ns(rundir)


def _cache_lock(cache: str):
    import fcntl

    os.makedirs(cache, exist_ok=True)
    f = open(os.path.join(cache, "layers.lock"), "a")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def _layer_hexes(store: str, key: str) -> list[str]:
    return [d["digest"].split(":", 1)[1] for d in image_record(store, key)["manifest"]["layers"]]


def pin(store: str, cache: str, key: str, tag: str) -> str:
    """Mark image `key` as used by `tag` and its layers as recently used; returns the pin file."""
    with _cache_lock(cache):
        d = os.path.join(cache, "pins")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{key}@{os.getpid()}@{tag}")
        open(p, "w").close()
        for h in _layer_hexes(store, key):
            s = os.path.join(cache, "sizes", h)
            if os.path.exists(s):
                os.utime(s)
    return p


def unpin(pin_file: str) -> None:
    try:
        os.remove(pin_file)
    except FileNotFoundError:
        pass


def _alive(pid: str) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _dir_bytes(path: str) -> int:
    n = 0
    for root, dirs, files in os.walk(path):
        for x in dirs + files:
            try:
                n += os.lstat(os.path.join(root, x)).st_blocks * 512
            except OSError:
                pass
    return n


def evict(store: str, cache: str, budget_bytes: float, max_age_s: float = 6 * 3600,
          keep_free_bytes: float = 30e9) -> dict:
    import shutil

    with _cache_lock(cache):
        keep, now = set(), time.time()
        pdir = os.path.join(cache, "pins")
        for name in (os.listdir(pdir) if os.path.isdir(pdir) else []):
            p = os.path.join(pdir, name)
            key, pid, _ = name.split("@", 2)
            if not _alive(pid) or now - os.path.getmtime(p) > max_age_s:
                unpin(p)
                continue
            try:
                keep.update(_layer_hexes(store, key))
            except OSError:
                pass
        ldir, sdir = os.path.join(cache, "layers"), os.path.join(cache, "sizes")
        os.makedirs(sdir, exist_ok=True)
        entries, total = [], 0
        for h in (os.listdir(ldir) if os.path.isdir(ldir) else []):
            if len(h) != 64:
                continue
            s = os.path.join(sdir, h)
            if not os.path.exists(s):
                with open(s, "w") as f:
                    f.write(str(_dir_bytes(os.path.join(ldir, h))))
            size = int(open(s).read() or 0)
            total += size
            entries.append((os.path.getmtime(s), h, size))
        free = shutil.disk_usage(cache).free
        budget_bytes = min(budget_bytes, max(0.0, total + free - keep_free_bytes))
        removed = []
        for _, h, size in sorted(entries):
            if total <= budget_bytes:
                break
            if h in keep:
                continue
            rmtree_ns(os.path.join(ldir, h))
            os.remove(os.path.join(sdir, h))
            total -= size
            removed.append(h)
        return {"bytes": total, "removed": len(removed), "pinned_layers": len(keep), "budget": budget_bytes}


def exec_cmd(store: str, key: str, cache: str, rundir: str, shell_cmd: str, timeout: float,
             extra_env: dict | None = None) -> tuple[int, str]:
    rec = image_record(store, key)
    cfg = rec["config"].get("config", {})
    env = list(cfg.get("Env") or ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"])
    env += ["HOME=/root", "HOSTNAME=swe-sandbox"] + [f"{k}={v}" for k, v in (extra_env or {}).items()]
    lower = prepare_layers(store, key, cache, log=lambda s: None)
    io = os.path.join(rundir, "io")
    os.makedirs(io, exist_ok=True)
    inner = ["/usr/bin/env", "-i", *env, "/bin/bash", "-c", shell_cmd]
    penv = dict(os.environ, SWE_RUN=rundir, SWE_LOWER=lower, SWE_IO=os.path.abspath(io))
    argv = ["unshare", "-Urmpf", "--kill-child", "--propagation", "private", "bash", "-c", _SH, "bash", *inner]
    try:
        p = subprocess.run(argv, env=penv, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or b"").decode("utf-8", "replace")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.swe.sandbox",
                                 description="Prepare an image's layers or run a command in its sandbox.")
    ap.add_argument("mode", choices=["layers", "run"])
    ap.add_argument("--store", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run")
    ap.add_argument("--io")
    ap.add_argument("--timeout", type=float)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("cmd", nargs="*")
    a = ap.parse_args(argv)
    if a.mode == "layers":
        print(prepare_layers(a.store, a.key, a.cache, log=lambda s: print(s, file=sys.stderr)))
        return
    sys.exit(run(a.store, a.key, a.cache, a.run, a.io, a.cmd or ["/bin/bash"], a.timeout, a.keep))


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
