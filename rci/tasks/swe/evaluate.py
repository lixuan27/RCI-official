"""Evaluate one SWE-bench Verified patch in the rootless sandbox with the official test specification."""
from __future__ import annotations

import argparse
import json
import os
import time

from rci import paths
from rci.tasks.swe import sandbox

DRIVER = r'''#!/bin/bash
cd /testbed
cp /swe_io/eval.sh /eval.sh
if [ -s /swe_io/patch.diff ]; then
  cp /swe_io/patch.diff /tmp/patch.diff
  applied=0; i=0
  for c in "git apply --verbose" "git apply --verbose --3way" "git apply --verbose --reject" "patch --batch --forward --fuzz=5 -p1 -i"; do
    [ $i -gt 0 ] && { git checkout -- . ; git clean -fd; } >> /swe_io/apply.log 2>&1
    i=$((i+1))
    if $c /tmp/patch.diff >> /swe_io/apply.log 2>&1; then applied=1; break; fi
  done
  if [ $applied = 0 ] && git apply --check --reverse /tmp/patch.diff >> /swe_io/apply.log 2>&1; then applied=1; fi
  echo "APPLIED=$applied ATTEMPTS=$i" >> /swe_io/apply.log
  [ $applied = 1 ] || exit 3
fi
git -c core.fileMode=false diff > /swe_io/diff_before.txt
t0=$(date +%s)
timeout "${SWE_EVAL_TIMEOUT:-1800}" /bin/bash /eval.sh > /swe_io/test_output.txt 2>&1
rc=$?
t1=$(date +%s)
echo "EVAL_RC=$rc EVAL_SECS=$((t1 - t0))" > /swe_io/eval_meta.txt
git -c core.fileMode=false diff > /swe_io/diff_after.txt
exit 0
'''


def load_instance(parquet: str, iid: str) -> dict:
    import pyarrow.parquet as pq
    for row in pq.read_table(parquet).to_pylist():
        if row["instance_id"] == iid:
            return row
    raise SystemExit(f"{iid} not in {parquet}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.swe.evaluate",
                                 description="Apply a patch to a SWE-bench Verified instance and run its tests.")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--patch-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default=None, help="default: $RCI_DATA/swe/oci")
    ap.add_argument("--cache", default=None, help="default: $RCI_CACHE/swe")
    ap.add_argument("--parquet", default=None, help="default: $RCI_DATA/swe/hf/verified.parquet")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args(argv)
    a.store = a.store or str(paths.data_dir("swe", "oci"))
    a.cache = a.cache or str(paths.cache_dir("swe"))
    a.parquet = a.parquet or str(paths.data_dir("swe", "hf", "verified.parquet"))
    from swebench.harness.grading import get_eval_report
    from swebench.harness.utils import make_test_spec

    inst = load_instance(a.parquet, a.instance)
    ts = make_test_spec(inst)
    patch = open(a.patch_file).read()
    out = {"instance_id": a.instance, "timing": {}}
    io = os.path.abspath(os.path.join(a.out, "io"))
    os.makedirs(io, exist_ok=True)
    for name, text in (("patch.diff", patch), ("eval.sh", ts.eval_script), ("driver.sh", DRIVER)):
        with open(os.path.join(io, name), "w") as f:
            f.write(text)
    t = time.time()
    rundir = os.path.join(a.cache, "runs", "eval", f"{a.instance}.{os.getpid()}.{int(t * 1000)}")
    rc = sandbox.run(os.path.abspath(a.store), a.instance, a.cache, rundir, io,
                     ["/bin/bash", "/swe_io/driver.sh"], timeout=a.timeout + 300,
                     extra_env={"SWE_EVAL_TIMEOUT": str(a.timeout)})
    out["timing"] = {"sandbox_s": round(time.time() - t, 1), "driver_rc": rc}
    tout = os.path.join(io, "test_output.txt")
    if os.path.exists(tout):
        pred = {"instance_id": a.instance, "model_name_or_path": "agent", "model_patch": patch}
        r = get_eval_report(test_spec=ts, prediction=pred, test_log_path=tout, include_tests_status=True)[a.instance]
        out["resolved"] = bool(r.get("resolved"))
        out["tests_status_counts"] = {k: {kk: len(vv) for kk, vv in v.items()}
                                      for k, v in r.get("tests_status", {}).items()}
    else:
        out["resolved"] = False
        out["error"] = "no test_output.txt (patch did not apply or the sandbox failed)"
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
