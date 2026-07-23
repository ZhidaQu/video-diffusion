#!/usr/bin/env python3
"""Build a markdown metric table for an explicit list of run tags on a benchmark."""
import argparse
import json
import os

from _report import METRICS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--tags", required=True, help="comma-separated run tags")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    name = os.path.basename(args.benchmark.rstrip("/"))
    lines = [f"# {name}", "", "| system | " + " | ".join(METRICS) + " |",
             "|" + "---|" * (len(METRICS) + 1)]
    for tag in args.tags.split(","):
        rj = os.path.join("runs", name, tag, "results.json")
        if not os.path.exists(rj):
            lines.append(f"| {tag} | (missing) |")
            continue
        a = json.load(open(rj))["aggregate"]
        lines.append("| " + tag + " | " + " | ".join(
            "n/a" if a.get(m) is None else f"{a[m]:.3f}" for m in METRICS) + " |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
