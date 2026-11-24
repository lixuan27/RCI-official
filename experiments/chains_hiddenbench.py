"""Figure 11a, Figure 10b and the rounds table: the gated HiddenBench chains T3 (with its matched single-agent
self-training chain) and T3k."""
from __future__ import annotations

import sys

from _chains import main
from _common import expand, shell

from rci import paths


def prepare(exp: dict, dry: bool) -> int:
    d = expand(exp["data"])
    out = str(paths.data_dir("chains", "hiddenbench"))
    w = d["written"]
    print("# 1. the base model writes hidden-profile and control tasks in the format of HiddenBench and admits them by")
    print("#    its own behaviour (serve the writer first: scripts/launch/serve.sh start " + d["writer"] + ")")
    print(f"RCI_DATA={w} " + shell(["scripts/data/prepare_splitbench.sh", "--writer", d["writer"], "--n", str(d["n"])]))
    print("# 2. freeze the selection set and the training pool once")
    print(shell(["python", "-m", "rci.chains.hiddenbench.tasks", "--hidden", f"{w}/splitbench/admission",
                 "--control", f"{w}/splitbench_control/admission", "--out", out,
                 "--agree", str(d["reference_agreement"])]))
    return 0


if __name__ == "__main__":
    sys.exit(main("chains_hiddenbench", extra={"prepare": prepare}))
