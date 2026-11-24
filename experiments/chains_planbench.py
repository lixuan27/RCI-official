"""Table 6, Figure 10a and the rounds table: the gated PlanBench (Blocksworld plan generation) chain."""
from __future__ import annotations

import json
import sys

from _chains import build, main


def prepare(exp: dict, dry: bool) -> int:
    ch = build(exp, "planbench")
    if dry:
        print(json.dumps({"sets_dir": str(ch.adapter.sets_dir()), "splits": ch.adapter.spec()}, default=str))
        return 0
    print(json.dumps(ch.adapter.prepare()))
    return 0


if __name__ == "__main__":
    sys.exit(main("chains_planbench", extra={"prepare": prepare}))
