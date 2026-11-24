"""Print the first port p >= BASE such that p .. p+N-1 are free on this host."""
from __future__ import annotations

import socket
import sys


def free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: free_port.py BASE [N=1] [LIMIT=32768]", file=sys.stderr)
        return 2
    base = int(argv[0])
    n = int(argv[1]) if len(argv) > 1 else 1
    limit = int(argv[2]) if len(argv) > 2 else 32768
    p = base
    while p + n <= limit:
        if all(free(p + i) for i in range(n)):
            print(p)
            return 0
        p += n
    print(f"no block of {n} free ports in [{base}, {limit})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
