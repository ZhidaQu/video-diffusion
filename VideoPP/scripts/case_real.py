#!/usr/bin/env python3
"""Real-anomaly case study on ShanghaiTech test (no GT-clean): detect -> remove.

Outputs a side-by-side video [anom | detected mask | per-frame | temporal] plus a
flicker curve and the frozen-scorer residual drop (anomaly reduced?).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import avr  # noqa: F401,E402
from avr.data.video_io import save_video  # noqa: E402
from avr.registry import DETECTORS, INPAINTERS, SCORERS  # noqa: E402
from avr.metrics.residual import residual_anomaly_drop  # noqa: E402

TEST = "data/raw/shanghaitech_test/SHANGHAI/SHANGHAI_Test"


def load_window(video, start, T, size):
    import cv2
    import imageio.v2 as imageio
    fdir = os.path.join(TEST, "frames", video)
    files = sorted(f for f in os.listdir(fdir) if f.endswith(".jpg"))
    frames = []
    for i in range(start, start + T):
        f = np.asarray(imageio.imread(os.path.join(fdir, files[i])))[..., :3]
        h = size
        w = int(round(f.shape[1] * size / f.shape[0] / 8) * 8)
        frames.append(cv2.resize(f, (w, h)))
    return np.stack(frames)


def anomaly_start(video, T):
    lab = np.load(os.path.join(TEST, "label", f"{video}.npy"))
    idx = np.where(lab > 0)[0]
    if len(idx) == 0:
        return 0
    c = int(idx[len(idx) // 2])
    return max(0, min(c - T // 2, len(lab) - T))


from _report import overlay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--prompt",
                    default="a bicycle. a car. a motorcycle. a person running. an unusual object.")
    ap.add_argument("--out", default="figures/case_real")
    args = ap.parse_args()

    start = anomaly_start(args.video, args.T)
    frames = load_window(args.video, start, args.T, args.size)
    print(f"{args.video} anomaly window @frame {start}, shape {frames.shape}")

    det = DETECTORS.build({"name": "grounded_sam2", "prompt": args.prompt,
                           "box_threshold": 0.30, "text_threshold": 0.25, "dilate": 7})
    mask = det.detect(frames)
    cover = [int(m.sum()) for m in mask]
    print("detected mask coverage/frame:", cover)

    inpaint = {
        "per-frame": {"name": "hf_diffusion", "steps": 25, "work_size": 512},
        "temporal": {"name": "hf_diffusion", "steps": 25, "work_size": 512,
                     "temporal": True, "blend": 0.5},
        "bg+temporal": {"name": "bg_diffusion", "steps": 25, "work_size": 512,
                        "temporal": True, "blend": 0.5},
    }
    restored = {k: INPAINTERS.build(dict(v)).inpaint(frames, mask)
                for k, v in inpaint.items()}

    scorer = SCORERS.build({"name": "clip_vad"})
    for k, r in restored.items():
        res = residual_anomaly_drop(scorer, frames, r, mask)
        print(f"  {k:10s} residual_drop={res['residual_drop']:+.3f}")

    panels = [frames, overlay(frames, mask)] + [restored[k] for k in inpaint]
    video = np.concatenate(panels, axis=2)
    save_video(args.out + ".mp4", video, fps=6)

    reg = mask
    fl = {k: [float(np.abs(r[t].astype(float) - r[t - 1])[reg[t]].mean())
              if reg[t].any() else 0.0 for t in range(1, args.T)]
          for k, r in restored.items()}
    plt.figure(figsize=(6, 3))
    for k, v in fl.items():
        plt.plot(range(1, args.T), v, marker=".", label=k)
    plt.title(f"{args.video}: per-frame flicker in removed region (lower=smoother)")
    plt.xlabel("frame"); plt.ylabel("L1"); plt.legend()
    plt.savefig(args.out + "_flicker.png", dpi=110, bbox_inches="tight")
    print("saved", args.out + ".mp4", "and", args.out + "_flicker.png")


if __name__ == "__main__":
    main()
