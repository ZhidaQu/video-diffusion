#!/usr/bin/env python3
"""Run all AVR systems against one benchmark and write a comparison table."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import avr  # noqa: F401,E402  (registers components)
from avr.runner import run_spec  # noqa: E402

from _report import METRICS, fmt  # noqa: E402

# Each system is (name, detector, inpainter). GT-mask systems isolate inpainting
# quality; heuristic/grounded systems are the full auto pipeline.
BASELINE_SYSTEMS = [
    ("naive_copy", {"name": "gt_mask"}, {"name": "copy_bg"}),
    ("classical_spatial", {"name": "gt_mask"}, {"name": "spatial_cv2"}),
    ("gt_flow", {"name": "gt_mask"}, {"name": "flow_fill"}),
    ("auto_heuristic", {"name": "heuristic_bgsub"}, {"name": "flow_fill"}),
]
REAL_SYSTEM = ("auto_grounded_diffusion",
               {"name": "grounded_sam", "box_threshold": 0.25,
                "text_threshold": 0.20, "dilate": 5},
               {"name": "hf_diffusion", "steps": 25, "work_size": 512})



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, help="prepared benchmark root")
    ap.add_argument("--real", action="store_true",
                    help="also run grounded_sam + hf_diffusion (GPU + downloads)")
    ap.add_argument("--scorer", default="clip_vad")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.benchmark, "manifest.json")):
        sys.exit(f"no benchmark at {args.benchmark} (run prepare_dataset.py first)")

    name = os.path.basename(args.benchmark.rstrip("/"))
    out_dir = os.path.join("runs", name)
    systems = list(BASELINE_SYSTEMS) + ([REAL_SYSTEM] if args.real else [])

    results = {}
    for sysname, det, inp in systems:
        cfg = {
            "run_id": f"{name}_{sysname}",
            "dataset": {"root": args.benchmark},
            "detector": det, "inpainter": inp, "scorer": {"name": args.scorer},
            "metrics": {"lpips": not args.no_lpips},
            "output": {"dir": os.path.join(out_dir, sysname), "save_videos": 2},
        }
        print(f"[run_all] {sysname}: {det['name']} + {inp['name']}")
        results[sysname] = run_spec(cfg, label=sysname)["aggregate"]

    header = "| system | " + " | ".join(METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 1)
    lines = [f"# AVR results — {name}", "", header, sep]
    for sysname, _, _ in systems:
        agg = results[sysname]
        lines.append("| " + sysname + " | "
                     + " | ".join(fmt(agg[m]) for m in METRICS) + " |")
    table = "\n".join(lines)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + table + f"\n\nwritten to {out_dir}/summary.md")


if __name__ == "__main__":
    main()
