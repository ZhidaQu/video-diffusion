#!/usr/bin/env python3
"""Build a qualitative case-study grid (rows=systems, cols=frames) for one clip."""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import avr  # noqa: F401,E402
from avr.data.datasets import load_dataset  # noqa: E402
from avr.pipeline import AVRPipeline  # noqa: E402
from avr.registry import DETECTORS, INPAINTERS  # noqa: E402
from real_worker import SYSTEMS  # noqa: E402

CELL = 180
LABEL_W = 150


from _report import overlay  # noqa: E402


def strip(frames, idxs):
    cells = []
    for i in idxs:
        img = Image.fromarray(frames[i]).resize((CELL, CELL))
        cells.append(np.asarray(img))
    return np.concatenate(cells, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--anomaly-type", default="moving_object")
    ap.add_argument("--systems", default="gt_flow,gt_diffusion,gt_diffusion_temporal")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clip = next(c for c in load_dataset(args.benchmark)
                if c.meta["anomaly_type"] == args.anomaly_type)
    T = clip.anom.shape[0]
    idxs = [0, T // 2, T - 1]

    rows = [("Anomalous", clip.anom),
            ("GT mask", overlay(clip.anom, clip.mask))]
    for name in args.systems.split(","):
        det, inp = SYSTEMS[name]
        if inp.get("name") == "hf_diffusion":
            inp = dict(inp, steps=25, work_size=512)
        pipe = AVRPipeline(DETECTORS.build(dict(det)), INPAINTERS.build(dict(inp)))
        out = pipe.run(clip.anom, gt_mask=clip.mask)
        rows.append((name, out["restored"]))
        if "grounded" in name:
            rows.append((name + " mask", overlay(clip.anom, out["mask"])))
    rows.append(("GT clean", clip.clean))

    grid_w = LABEL_W + CELL * len(idxs)
    canvas = Image.new("RGB", (grid_w, CELL * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (label, frames) in enumerate(rows):
        canvas.paste(Image.fromarray(strip(frames, idxs)), (LABEL_W, r * CELL))
        draw.text((6, r * CELL + CELL // 2), label, fill="black")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    canvas.save(args.out)
    print("saved", args.out, f"({clip.clip_id}, frames {idxs})")


if __name__ == "__main__":
    main()
