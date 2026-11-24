"""Figure mechanism (a), Proposition 2: the confidence above which one more claim raises the expected score."""
from __future__ import annotations

import argparse
import sys


def recall_gain(h: int, r: int, t: int, q: float) -> float:
    return q / t


def f1_gain(h: int, r: int, t: int, q: float) -> float:
    return 2 * (h + q) / (r + t + 1) - 2 * h / (r + t)


def gated_gain(h: int, r: int, t: int, q: float, c: float) -> float:
    return (q - c * (1 - q)) / t


def threshold_f1(f1: float) -> float:
    return f1 / 2


def threshold_gated(c: float) -> float:
    return c / (1 + c)


def check(t: int = 20, grid: int = 400) -> int:
    bad = 0
    for r in range(0, 2 * t + 1):
        for h in range(0, min(r, t) + 1):
            if r + t == 0:
                continue
            f1 = 2 * h / (r + t)
            for k in range(1, grid):
                q = k / grid
                if abs(q - threshold_f1(f1)) > 1e-9 and (f1_gain(h, r, t, q) > 0) != (q > threshold_f1(f1)):
                    bad += 1
                for c in (0.25, 0.5, 1.0):
                    if abs(q - threshold_gated(c)) > 1e-9 and (gated_gain(h, r, t, q, c) > 0) != (q > threshold_gated(c)):
                        bad += 1
                if recall_gain(h, r, t, q) <= 0:
                    bad += 1
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reporting thresholds of Proposition 2 (Figure mechanism a).")
    ap.add_argument("--plot", default=None, help="write the figure to this file (needs matplotlib)")
    ap.add_argument("--check", action="store_true", help="verify the closed forms against a scan of q")
    a = ap.parse_args(argv)
    if a.check:
        bad = check()
        print(f"closed forms {'hold' if bad == 0 else 'fail'} ({bad} mismatches)")
        if bad:
            return 1
    print("F1    recall  F1-score  gated c=0.25  gated c=0.5")
    for i in range(0, 13):
        f1 = i * 0.05
        print(f"{f1:4.2f}  {0.0:6.3f}  {threshold_f1(f1):8.3f}  {threshold_gated(0.25):12.3f}  {threshold_gated(0.5):11.3f}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [i / 200 * 0.6 for i in range(201)]
        fig, ax = plt.subplots(figsize=(3.35, 2.3))
        ax.plot(xs, [0.0] * len(xs), lw=1.4, label="recall")
        ax.plot(xs, [threshold_f1(x) for x in xs], lw=1.4, label="F1: q* = F1/2")
        ax.plot(xs, [threshold_gated(0.25)] * len(xs), lw=1.8, label="gated, c = 0.25")
        ax.plot(xs, [threshold_gated(0.5)] * len(xs), lw=1.1, ls="--", label="gated, c = 0.5")
        ax.set_xlabel("current exact-pair F1 of the group")
        ax.set_ylabel("reporting threshold q*")
        ax.set_xlim(0, 0.6)
        ax.set_ylim(-0.02, 0.39)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(a.plot)
        print(f"wrote {a.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
