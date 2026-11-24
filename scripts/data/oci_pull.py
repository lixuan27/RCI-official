#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])
SEG = 16 * 1024 * 1024
SINGLE_MAX = 8 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


UA = "docker/24.0.7 go/go1.20.10 os/linux arch/amd64"
_opener = urllib.request.build_opener(_NoRedirect)
_opener.addheaders = [("User-Agent", UA)]
_default = urllib.request.build_opener()
_default.addheaders = [("User-Agent", UA)]
urllib.request.install_opener(_default)


def parse_ref(ref: str):
    tag, name = "latest", ref
    if "@" in name:
        name, tag = name.split("@", 1)
    else:
        last = name.rsplit("/", 1)[-1]
        if ":" in last:
            name, tag = name.rsplit(":", 1)
    first = name.split("/", 1)[0]
    if "." in first or ":" in first or first == "localhost":
        registry, repo = name.split("/", 1)
    else:
        registry, repo = "registry-1.docker.io", name
    if registry in ("docker.io", "index.docker.io"):
        registry = "registry-1.docker.io"
    if registry == "registry-1.docker.io" and "/" not in repo:
        repo = "library/" + repo
    return registry, repo, tag


class Registry:
    def __init__(self, registry: str, repo: str, timeout: float = 60.0):
        self.base = f"https://{registry}/v2/{repo}"
        self.registry, self.repo, self.timeout = registry, repo, timeout
        self.token = None
        self.lock = threading.Lock()

    def _auth(self, www: str):
        m = dict(re.findall(r'(\w+)="([^"]*)"', www))
        realm = m.pop("realm", None)
        if not realm:
            raise RuntimeError(f"cannot parse WWW-Authenticate: {www!r}")
        m.setdefault("scope", f"repository:{self.repo}:pull")
        url = realm + "?" + urllib.parse.urlencode(m)
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            d = json.load(r)
        self.token = d.get("token") or d.get("access_token")

    def open(self, path: str, headers=None, _retry=True):
        h = dict(headers or {})
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base + path, headers=h)
        try:
            return _opener.open(req, timeout=self.timeout), None
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                loc = urllib.parse.urljoin(self.base + path, e.headers.get("Location"))
                return None, loc
            if e.code == 401 and _retry:
                with self.lock:
                    self._auth(e.headers.get("WWW-Authenticate", ""))
                return self.open(path, headers, _retry=False)
            raise

    def manifest(self, ref: str):
        r, loc = self.open(f"/manifests/{ref}", {"Accept": ACCEPT})
        if r is None:
            r = urllib.request.urlopen(urllib.request.Request(loc, headers={"Accept": ACCEPT}),
                                       timeout=self.timeout)
        with r:
            body = r.read()
            hdr = r.headers.get("Docker-Content-Digest")
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        if hdr and hdr != digest:
            raise RuntimeError(f"manifest header digest {hdr} != body sha256 {digest}")
        if ref.startswith("sha256:") and ref != digest:
            raise RuntimeError(f"asked for manifest {ref}, body hashes to {digest}")
        return json.loads(body), digest, body

    def blob_url(self, digest: str):
        r, loc = self.open(f"/blobs/{digest}", {"Range": "bytes=0-0"})
        if r is not None:
            r.close()
            return self.base + f"/blobs/{digest}", {"Authorization": f"Bearer {self.token}"} if self.token else {}
        return loc, {}


def _get_range(url, headers, start, end, dest, timeout=120, allow_full=False):
    want = end - start + 1
    h = dict(headers)
    h["Range"] = f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        full = r.status == 200 and allow_full
        if r.status != 206 and not full:
            raise IOError(f"expected 206, got {r.status}")
        skip = start if full else 0
        while skip > 0:
            chunk = r.read(min(1 << 20, skip))
            if not chunk:
                raise IOError(f"body ended {skip} bytes before offset {start}")
            skip -= len(chunk)
        tmp = dest + ".tmp"
        n = 0
        with open(tmp, "wb") as f:
            while not full or n < want:
                chunk = r.read(min(1 << 20, want - n) if full else 1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                n += len(chunk)
    if n != want:
        os.remove(tmp)
        raise IOError(f"short segment {n}/{want}")
    os.replace(tmp, dest)
    return n


def fetch_blob(reg: Registry, digest: str, size: int, dest: str, pool: cf.ThreadPoolExecutor,
               tries: int = 8) -> int:
    hexd = digest.split(":", 1)[1]
    if size <= SINGLE_MAX:
        segs = [(0, size - 1)] if size > 0 else []
    else:
        segs = [(s, min(s + SEG, size) - 1) for s in range(0, size, SEG)]
    segdir = dest + ".segs"
    os.makedirs(segdir, exist_ok=True)
    url_box = {"url": None, "hdr": None, "t": 0.0}
    url_lock = threading.Lock()

    def current_url(force=False):
        with url_lock:
            if force or url_box["url"] is None or time.time() - url_box["t"] > 300:
                url_box["url"], url_box["hdr"] = reg.blob_url(digest)
                url_box["t"] = time.time()
            return url_box["url"], url_box["hdr"]

    def one(i, s, e):
        p = os.path.join(segdir, f"{i:05d}")
        if os.path.exists(p) and os.path.getsize(p) == e - s + 1:
            return 0
        err = None
        for attempt in range(tries):
            try:
                url, hdr = current_url(force=attempt > 0)
                return _get_range(url, hdr, s, e, p, allow_full=attempt >= 2)
            except Exception as ex:
                err = ex
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"segment {i} of {hexd[:12]} failed: {err!r}")

    got = sum(f.result() for f in [pool.submit(one, i, s, e) for i, (s, e) in enumerate(segs)])
    h = hashlib.sha256()
    tmp = dest + ".part"
    with open(tmp, "wb") as out:
        for i in range(len(segs)):
            with open(os.path.join(segdir, f"{i:05d}"), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
                    out.write(chunk)
    if h.hexdigest() != hexd or os.path.getsize(tmp) != size:
        os.remove(tmp)
        for fn in os.listdir(segdir):
            os.remove(os.path.join(segdir, fn))
        raise RuntimeError(f"blob {hexd[:12]} sha256/size mismatch; segments discarded")
    os.replace(tmp, dest)
    for fn in os.listdir(segdir):
        os.remove(os.path.join(segdir, fn))
    os.rmdir(segdir)
    return got


def resolve(ref: str, platform=("linux", "amd64")):
    registry, repo, tag = parse_ref(ref)
    reg = Registry(registry, repo)
    man, mdigest, body = reg.manifest(tag)
    if "manifests" in man:
        cands = [m for m in man["manifests"]
                 if m.get("platform", {}).get("os") == platform[0]
                 and m.get("platform", {}).get("architecture") == platform[1]]
        if not cands:
            raise RuntimeError(f"{ref}: no {platform} manifest")
        mdigest = cands[0]["digest"]
        man, _, body = reg.manifest(mdigest)
    return reg, (registry, repo, tag), man, mdigest, body


def pull(store: str, ref: str, key: str, pool: cf.ThreadPoolExecutor,
         blob_locks: dict, blob_locks_lock: threading.Lock) -> dict:
    out = os.path.join(store, "images", key + ".json")
    if os.path.exists(out):
        with open(out) as f:
            return json.load(f)
    reg, (registry, repo, tag), man, mdigest, mbody = resolve(ref)
    blobdir = os.path.join(store, "blobs", "sha256")
    os.makedirs(blobdir, exist_ok=True)
    mdir = os.path.join(store, "manifests", "sha256")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, mdigest.split(":", 1)[1]), "wb") as f:
        f.write(mbody)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0, nbytes, nnew = time.time(), 0, 0
    for desc in [man["config"]] + man["layers"]:
        hexd = desc["digest"].split(":", 1)[1]
        dest = os.path.join(blobdir, hexd)
        with blob_locks_lock:
            lk = blob_locks.setdefault(hexd, threading.Lock())
        with lk:
            if os.path.exists(dest):
                continue
            nbytes += fetch_blob(reg, desc["digest"], desc["size"], dest, pool)
            nnew += 1
    with open(os.path.join(blobdir, man["config"]["digest"].split(":", 1)[1])) as f:
        config = json.load(f)
    rec = {"ref": ref, "registry": registry, "repo": repo, "tag": tag,
           "manifest_digest": mdigest, "manifest": man, "config": config,
           "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, out)
    dt = time.time() - t0
    total = sum(d.get("size", 0) for d in man["layers"])
    print(f"OK {key}: {len(man['layers'])} layers {total/1e6:.0f} MB; fetched {nnew} blobs "
          f"{nbytes/1e6:.0f} MB in {dt:.0f}s ({nbytes/1e6/max(dt, 1e-3):.1f} MB/s) "
          f"manifest={mdigest}", flush=True)
    return rec


def official_digest(image: str, platform=("linux", "amd64")) -> dict:
    registry, repo, tag = parse_ref(image)
    if registry != "registry-1.docker.io":
        raise ValueError(f"{image}: not a Docker Hub reference")
    op = urllib.request.build_opener()
    op.addheaders = [("User-Agent", UA)]
    tok_url = ("https://auth.docker.io/token?service=registry.docker.io&scope="
               + urllib.parse.quote(f"repository:{repo}:pull"))
    with op.open(tok_url, timeout=30) as r:
        tok = json.load(r)["token"]
    url = f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}"
    hdr = {"Authorization": f"Bearer {tok}", "Accept": ACCEPT}
    with op.open(urllib.request.Request(url, method="HEAD", headers=hdr), timeout=30) as r:
        top, ctype = r.headers["Docker-Content-Digest"], r.headers.get("Content-Type", "")
    out = {"image": image, "tag_digest": top, "media_type": ctype,
           "source": "registry-1.docker.io"}
    if "index" in ctype or "manifest.list" in ctype:
        with op.open(urllib.request.Request(url, headers=hdr), timeout=30) as r:
            body = r.read()
        if "sha256:" + hashlib.sha256(body).hexdigest() != top:
            raise RuntimeError(f"{image}: Docker Hub index body does not hash to {top}")
        kids = [m["digest"] for m in json.loads(body)["manifests"]
                if m.get("platform", {}).get("os") == platform[0]
                and m.get("platform", {}).get("architecture") == platform[1]]
        if not kids:
            raise RuntimeError(f"{image}: no {platform} manifest on Docker Hub")
        out["platform_digest"] = kids[0]
    else:
        out["platform_digest"] = top
    return out


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def verify_image(store: str, key: str, official: dict) -> dict:
    with open(os.path.join(store, "images", key + ".json")) as f:
        rec = json.load(f)
    res = {"official": official, "used_manifest_digest": rec["manifest_digest"],
           "ref": rec["ref"], "problems": []}
    want = official.get("platform_digest")
    mpath = os.path.join(store, "manifests", "sha256", rec["manifest_digest"].split(":", 1)[1])
    if not os.path.exists(mpath):
        res["problems"].append("raw manifest bytes missing")
    else:
        got = _sha256_file(mpath)
        if got != want or rec["manifest_digest"] != want:
            res["problems"].append(f"manifest {got} != Docker Hub {want}")
        with open(mpath, "rb") as f:
            if json.loads(f.read()) != rec["manifest"]:
                res["problems"].append("recorded manifest differs from raw bytes")
    descs = [rec["manifest"]["config"]] + rec["manifest"]["layers"]
    nok = 0
    for d in descs:
        p = os.path.join(store, "blobs", "sha256", d["digest"].split(":", 1)[1])
        if not os.path.exists(p):
            res["problems"].append(f"missing blob {d['digest']}")
        elif os.path.getsize(p) != d["size"] or _sha256_file(p) != d["digest"]:
            res["problems"].append(f"bad blob {d['digest']}")
        else:
            nok += 1
    res["blobs_verified"] = f"{nok}/{len(descs)}"
    res["passed"] = not res["problems"]
    return res


def main():
    ap = argparse.ArgumentParser(description="Pull container images into a content-addressed store.")
    ap.add_argument("--store")
    ap.add_argument("--ref")
    ap.add_argument("--key")
    ap.add_argument("--list", help="TSV of '<key>\\t<ref>' lines")
    ap.add_argument("--jobs", type=int, default=16, help="concurrent HTTP connections")
    ap.add_argument("--images", type=int, default=2, help="images pulled concurrently")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--official-out", help="write Docker Hub digests for the refs (official names) to this JSON")
    a = ap.parse_args()
    if a.list:
        with open(a.list) as f:
            jobs = [tuple(l.rstrip("\n").split("\t")[:2]) for l in f if l.strip()]
    else:
        jobs = [(a.key or a.ref.replace("/", "_"), a.ref)]
    if a.official_out:
        recs = {}
        if os.path.exists(a.official_out):
            with open(a.official_out) as f:
                recs = json.load(f)
        for key, ref in jobs:
            try:
                recs[key] = official_digest(ref)
                print(key, recs[key]["platform_digest"], recs[key]["media_type"], flush=True)
            except Exception as e:
                print(f"FAIL {key} {ref}: {e!r}", flush=True)
        tmp = a.official_out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(recs, f, indent=1)
        os.replace(tmp, a.official_out)
        return
    if a.manifest_only:
        for key, ref in jobs:
            try:
                _, _, man, d, _ = resolve(ref)
                print(key, d, sum(l["size"] for l in man["layers"]),
                      json.dumps([(l["digest"][7:19], l["size"]) for l in man["layers"]]), flush=True)
            except Exception as e:
                print(f"FAIL {key} {ref}: {e!r}", flush=True)
        return
    pool = cf.ThreadPoolExecutor(a.jobs)
    locks, locks_lock = {}, threading.Lock()
    fails = 0
    with cf.ThreadPoolExecutor(a.images) as ipool:
        futs = {ipool.submit(pull, a.store, ref, key, pool, locks, locks_lock): key for key, ref in jobs}
        for f in cf.as_completed(futs):
            try:
                f.result()
            except Exception as e:
                fails += 1
                print(f"FAIL {futs[f]}: {e!r}", file=sys.stderr, flush=True)
    pool.shutdown()
    print(f"done: {len(jobs) - fails}/{len(jobs)} images ok", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
