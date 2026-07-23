#!/usr/bin/env python3
"""Collate runs/<bench>/<system>/results.json into runs/overnight/MASTER.{md,json}."""
import glob
import json
import os
import sys

from _report import METRICS  # noqa: E402
ORDER = ["naive_copy", "classical_spatial", "gt_flow", "auto_heuristic",
         "gt_diffusion", "gt_diffusion_temporal", "gt_bg",
         "gt_diffusion_t03", "gt_diffusion_t07",
         "grounded_diffusion", "grounded2_diffusion", "grounded2_bg"]


def main():
    benches = sys.argv[1:]
    out = ["# Overnight results", ""]
    data = {}
    for bench in benches:
        name = os.path.basename(bench.rstrip("/"))
        systems = {}
        for rj in glob.glob(f"runs/{name}/*/results.json"):
            s = os.path.basename(os.path.dirname(rj))
            systems[s] = json.load(open(rj))["aggregate"]
        data[name] = systems
        out += [f"## {name}", "", "| system | " + " | ".join(METRICS) + " |",
                "|" + "---|" * (len(METRICS) + 1)]
        keys = [k for k in ORDER if k in systems] + \
               [k for k in systems if k not in ORDER]
        for s in keys:
            a = systems[s]
            out.append("| " + s + " | " + " | ".join(
                "n/a" if a[m] is None else f"{a[m]:.3f}" for m in METRICS) + " |")
        out.append("")
    os.makedirs("runs/overnight", exist_ok=True)
    open("runs/overnight/MASTER.md", "w").write("\n".join(out) + "\n")
    json.dump(data, open("runs/overnight/MASTER.json", "w"), indent=2)
    print("collated", sum(len(v) for v in data.values()), "systems across",
          len(data), "benchmarks")


if __name__ == "__main__":
    main()
