#!/usr/bin/env python3
"""Build an AVR benchmark by injecting anomalies into real normal clips."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avr.data.datasets import build_benchmark  # noqa: E402
from avr.data.real_sources import SOURCES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SOURCES), required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="")
    ap.add_argument("--clips", type=int, default=60)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="tiny/fast: 6 clips, T=8, size=128")
    args = ap.parse_args()

    if args.quick:
        args.clips, args.T, args.size = 6, 8, 128
    out = args.out or f"data/{args.dataset}_avr"
    stride = args.stride or None

    clips = SOURCES[args.dataset](args.raw, T=args.T, size=args.size,
                                  stride=stride, limit=args.clips)
    build_benchmark(out, clips)
    n = len(os.listdir(out)) - 1  # minus manifest.json
    print(f"built {n} clips at {out}")


if __name__ == "__main__":
    main()
