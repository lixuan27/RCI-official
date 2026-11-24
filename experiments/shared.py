"""Table 1: GPQA-Diamond and DocSweep with four 27B agents on the equal-budget ladder."""
from __future__ import annotations

import sys

from _common import dispatch, load, parser, shell

from rci import paths


def main(argv=None) -> int:
    ap, sub, common = parser("Shared information (GPQA-Diamond) and a shared corpus (DocSweep), Table 1.")
    p = sub.add_parser("prompts", parents=[common], help="print the commands that write the DocSweep prompts A and B")
    p.add_argument("--base-urls", default=None)
    a = ap.parse_args(argv)
    exp = load("shared")
    got = dispatch(exp, a)
    if got is not None:
        return got
    for which in ("A", "B"):
        print(shell(["python", "-m", "rci.organize.self_prompt", "docsweep", "rules", "--which", which, "--base-urls",
                     a.base_urls or "$RCI_BASE_URLS", "--model", exp["models"]["q27"], "--out",
                     str(paths.results_dir("shared", "notes", f"docsweep_rules_{which}.json"))]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
