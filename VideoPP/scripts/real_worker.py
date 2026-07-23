#!/usr/bin/env python3
"""Run one real system on the currently visible GPU. Launched by run_real_dual.py."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import avr  # noqa: F401,E402  (registers components)
from avr.runner import run_spec  # noqa: E402

GROUNDED = {"name": "grounded_sam", "box_threshold": 0.25,
            "text_threshold": 0.20, "dilate": 5}
GROUNDED2 = {"name": "grounded_sam2", "box_threshold": 0.25,
             "text_threshold": 0.20, "dilate": 5}
GT = {"name": "gt_mask"}
HEUR = {"name": "heuristic_bgsub"}


def _diff(temporal=False, blend=0.5):
    d = {"name": "hf_diffusion"}
    if temporal:
        d.update(temporal=True, blend=blend)
    return d


def _bg(blend=0.5):
    return {"name": "bg_diffusion", "temporal": True, "blend": blend}


SYSTEMS = {
    # classical baselines (fast)
    "naive_copy": (GT, {"name": "copy_bg"}),
    "classical_spatial": (GT, {"name": "spatial_cv2"}),
    "gt_flow": (GT, {"name": "flow_fill"}),
    "auto_heuristic": (HEUR, {"name": "flow_fill"}),
    # diffusion, GT mask
    "gt_diffusion": (GT, _diff()),
    "gt_diffusion_temporal": (GT, _diff(True, 0.5)),
    "gt_bg": (GT, _bg()),
    "gt_diffusion_t03": (GT, _diff(True, 0.3)),
    "gt_diffusion_t07": (GT, _diff(True, 0.7)),
    # diffusion, automatic mask
    "grounded_diffusion": (GROUNDED, _diff()),
    "grounded2_diffusion": (GROUNDED2, _diff()),
    "grounded2_bg": (GROUNDED2, _bg()),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--system", choices=list(SYSTEMS), required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--work-size", type=int, default=512)
    ap.add_argument("--scorer", default="clip_vad")
    args = ap.parse_args()

    det, inp = SYSTEMS[args.system]
    if inp.get("name") in ("hf_diffusion", "bg_diffusion"):
        inp = dict(inp, steps=args.steps, work_size=args.work_size)
    name = os.path.basename(args.benchmark.rstrip("/"))
    cfg = {
        "run_id": f"{name}_{args.system}",
        "dataset": {"root": args.benchmark},
        "detector": det, "inpainter": inp, "scorer": {"name": args.scorer},
        "metrics": {"lpips": True},
        "output": {"dir": os.path.join("runs", name, args.system),
                   "save_videos": 3},
    }
    r = run_spec(cfg, label=args.system)
    print(f"[{args.system}] aggregate: {r['aggregate']}")


if __name__ == "__main__":
    main()
