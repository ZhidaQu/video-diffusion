#!/usr/bin/env python3
"""Case study that isolates the method's advantage: error-vs-GT heatmaps and
per-frame error / flicker curves (temporal should be lowest and smoothest)."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import avr  # noqa: F401,E402
from avr.data.datasets import load_dataset  # noqa: E402
from avr.pipeline import AVRPipeline  # noqa: E402
from avr.registry import DETECTORS, INPAINTERS  # noqa: E402
from real_worker import SYSTEMS  # noqa: E402


def region_of(mask, dilate=9):
    import cv2
    k = np.ones((dilate, dilate), np.uint8)
    return np.stack([cv2.dilate(m.astype(np.uint8), k).astype(bool) for m in mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--anomaly-type", default="moving_object")
    ap.add_argument("--systems", default="gt_flow,gt_diffusion,gt_diffusion_temporal")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clip = next(c for c in load_dataset(args.benchmark)
                if c.meta["anomaly_type"] == args.anomaly_type)
    clean = clip.clean.astype(np.float32)
    region = region_of(clip.mask)
    T = clip.anom.shape[0]
    mid = T // 2
    names = args.systems.split(",")

    restored, errcurve, flicker, heat = {}, {}, {}, {}
    for name in names:
        det, inp = SYSTEMS[name]
        if inp.get("name") == "hf_diffusion":
            inp = dict(inp, steps=25, work_size=512)
        pipe = AVRPipeline(DETECTORS.build(dict(det)), INPAINTERS.build(dict(inp)))
        r = pipe.run(clip.anom, gt_mask=clip.mask)["restored"].astype(np.float32)
        restored[name] = r
        errcurve[name] = [float(np.abs(r[t] - clean[t])[region[t]].mean())
                          for t in range(T)]
        flicker[name] = [float(np.abs(r[t] - r[t - 1])[region[t]].mean())
                         for t in range(1, T)]
        heat[name] = np.abs(r[mid] - clean[mid]).mean(-1)

    vmax = max(h.max() for h in heat.values())
    fig = plt.figure(figsize=(3 * len(names), 8))
    gs = fig.add_gridspec(2, len(names), height_ratios=[1.3, 1])
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(heat[name], cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"{name}\n|restored-GT| mean={np.mean(errcurve[name]):.2f}",
                     fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.01, label="|error| (mid frame)")

    axc = fig.add_subplot(gs[1, :2] if len(names) >= 2 else gs[1, 0])
    for name in names:
        axc.plot(errcurve[name], label=name, marker=".")
    axc.set_title("per-frame error vs GT (in anomaly region)")
    axc.set_xlabel("frame"); axc.set_ylabel("L1"); axc.legend(fontsize=8)

    if len(names) >= 2:
        axf = fig.add_subplot(gs[1, 2:] if len(names) > 2 else gs[1, 1])
        for name in names:
            axf.plot(range(1, T), flicker[name], label=name, marker=".")
        axf.set_title("per-frame flicker |Δ| (lower = smoother)")
        axf.set_xlabel("frame"); axf.set_ylabel("L1"); axf.legend(fontsize=8)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print("saved", args.out, f"({clip.clip_id})")
    for name in names:
        print(f"  {name:28s} err={np.mean(errcurve[name]):.3f} "
              f"flicker={np.mean(flicker[name]):.3f}")


if __name__ == "__main__":
    main()
