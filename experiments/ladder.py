from __future__ import annotations

import sys

from _common import dispatch, load, parser


def main(argv=None) -> int:
    ap, _, _ = parser("HiddenBench ladder of Table 2 and Figure 3b.")
    a = ap.parse_args(argv)
    return dispatch(load("ladder"), a)


if __name__ == "__main__":
    sys.exit(main())
