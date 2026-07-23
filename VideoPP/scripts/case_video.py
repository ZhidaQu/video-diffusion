#!/usr/bin/env python3
"""Side-by-side case-study video: panels [anom | mask | systems... | GT] over a clip."""
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
from avr.data.video_io import save_video  # noqa: E402
from avr.pipeline import AVRPipeline  # noqa: E402
from avr.registry import DETECTORS, INPAINTERS  # noqa: E402
from real_worker import SYSTEMS  # noqa: E402

CELL = 256


from _report import overlay  # noqa: E402


def labeled(frames, text):
    out = []
    for f in frames:
        img = Image.fromarray(f).resize((CELL, CELL)).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, CELL, 16], fill="black")
        d.text((4, 3), text, fill="white")
        out.append(np.asarray(img))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--anomaly-type", default="moving_object")
    ap.add_argument("--systems", default="gt_diffusion,gt_diffusion_temporal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    clip = next(c for c in load_dataset(args.benchmark)
                if c.meta["anomaly_type"] == args.anomaly_type)
    panels = [labeled(clip.anom, "anomalous"),
              labeled(overlay(clip.anom, clip.mask), "GT mask")]
    for name in args.systems.split(","):
        det, inp = SYSTEMS[name]
        if inp.get("name") == "hf_diffusion":
            inp = dict(inp, steps=25, work_size=512)
        pipe = AVRPipeline(DETECTORS.build(dict(det)), INPAINTERS.build(dict(inp)))
        restored = pipe.run(clip.anom, gt_mask=clip.mask)["restored"]
        panels.append(labeled(restored, name))
    panels.append(labeled(clip.clean, "GT clean"))

    video = np.concatenate(panels, axis=2)  # concat along width
    save_video(args.out, video, fps=args.fps)
    print("saved", args.out, f"({clip.clip_id}, {video.shape[0]} frames, "
          f"{len(panels)} panels)")


if __name__ == "__main__":
    main()
