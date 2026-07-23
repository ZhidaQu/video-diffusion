#!/usr/bin/env python3
"""Run one (detector, inpainter, scorer) config on a benchmark.

Writes metrics to runs/<benchmark>/<tag>/. Detector/inpainter parameters are
supplied as flags so experiment scripts can sweep them.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import avr  # noqa: F401,E402
from avr.runner import run_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--inpainter", required=True)
    ap.add_argument("--scorer", default="clip_vad")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--box-threshold", type=float)
    ap.add_argument("--text-threshold", type=float)
    ap.add_argument("--dilate", type=int)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--work-size", type=int, default=512)
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument("--save-videos", type=int, default=2)
    args = ap.parse_args()

    det = {"name": args.detector}
    for k in ("box_threshold", "text_threshold", "dilate"):
        v = getattr(args, k)
        if v is not None:
            det[k] = v
    inp = {"name": args.inpainter}
    if args.inpainter in ("hf_diffusion", "bg_diffusion"):
        inp["steps"] = args.steps
        inp["work_size"] = args.work_size
        inp["temporal"] = args.temporal
        inp["blend"] = args.blend

    name = os.path.basename(args.benchmark.rstrip("/"))
    cfg = {
        "run_id": f"{name}_{args.tag}",
        "dataset": {"root": args.benchmark},
        "detector": det, "inpainter": inp, "scorer": {"name": args.scorer},
        "metrics": {"lpips": True, "fvd": True},
        "output": {"dir": os.path.join("runs", name, args.tag),
                   "save_videos": args.save_videos},
    }
    r = run_spec(cfg, label=args.tag)
    print(f"[{args.tag}] {r['aggregate']}")


if __name__ == "__main__":
    main()
