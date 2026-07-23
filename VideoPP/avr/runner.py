"""Run a config over the benchmark and write metrics, per-clip CSV, and sample videos."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np

from .config import load_config
from .data.datasets import load_dataset
from .data.video_io import save_triptych
from .metrics import LPIPS, mask_iou, psnr, ssim, warp_error
from .metrics.fvd import FVD
from .metrics.residual import residual_anomaly_drop
from .pipeline import AVRPipeline
from .registry import DETECTORS, INPAINTERS, SCORERS

DEFAULT_SCORER = {"name": "clip_vad"}


def _ensure_dataset(cfg: dict) -> str:
    root = cfg["dataset"]["root"]
    if not os.path.exists(os.path.join(root, "manifest.json")):
        raise FileNotFoundError(
            f"No benchmark at {root}; build one with scripts/prepare_dataset.py")
    return root


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def run_config(config_path: str) -> Dict:
    return run_spec(load_config(config_path), label=config_path)


def run_spec(cfg: dict, label: str = "run") -> Dict:
    run_id = cfg.get("run_id") or "run"
    out_dir = cfg.get("output", {}).get("dir", os.path.join("runs", run_id))
    os.makedirs(out_dir, exist_ok=True)
    n_save = int(cfg.get("output", {}).get("save_videos", 3))
    use_lpips = bool(cfg.get("metrics", {}).get("lpips", True))

    root = _ensure_dataset(cfg)
    detector = DETECTORS.build(cfg["detector"])
    inpainter = INPAINTERS.build(cfg["inpainter"])
    scorer = SCORERS.build(cfg.get("scorer", DEFAULT_SCORER))
    pipe = AVRPipeline(detector, inpainter)
    lpips_fn = LPIPS(**cfg.get("metrics", {}).get("lpips_args", {})) if use_lpips else None

    rows = []
    by_type = defaultdict(list)
    restored_all, clean_all = [], []
    saved = 0
    for clip in load_dataset(root):
        out = pipe.run(clip.anom, gt_mask=clip.mask)
        restored, pred_mask = out["restored"], out["mask"]
        restored_all.append(restored)
        clean_all.append(clip.clean)

        lp = lpips_fn(restored, clip.clean) if lpips_fn is not None else None
        tlp = (lpips_fn(restored[:-1], restored[1:])
               if lpips_fn is not None and len(restored) > 1 else None)
        res = residual_anomaly_drop(scorer, clip.anom, restored, clip.mask)
        row = {
            "clip_id": clip.clip_id,
            "anomaly_type": clip.meta["anomaly_type"],
            "psnr": psnr(restored, clip.clean),
            "ssim": ssim(restored, clip.clean),
            "lpips": lp,
            "tlp": tlp,
            "warp_error": warp_error(restored),
            "residual_drop": res["residual_drop"],
            "mask_iou": mask_iou(pred_mask, clip.mask),
        }
        rows.append(row)
        by_type[row["anomaly_type"]].append(row)

        if saved < n_save:
            save_triptych(os.path.join(out_dir, "samples", f"{clip.clip_id}.mp4"),
                          clip.anom, restored, clip.clean)
            saved += 1

    metric_keys = ["psnr", "ssim", "lpips", "tlp", "warp_error",
                   "residual_drop", "mask_iou"]
    agg = {k: _mean([r[k] for r in rows]) for k in metric_keys}
    if cfg.get("metrics", {}).get("fvd", True):
        agg["fvd"] = FVD(**cfg.get("metrics", {}).get("fvd_args", {})).compute(
            restored_all, clean_all)
    per_type = {
        t: {k: _mean([r[k] for r in rs]) for k in metric_keys}
        for t, rs in by_type.items()
    }
    result = {
        "run_id": run_id,
        "config": label,
        "n_clips": len(rows),
        "components": {
            "detector": cfg["detector"], "inpainter": cfg["inpainter"],
            "scorer": cfg.get("scorer", DEFAULT_SCORER),
        },
        "aggregate": agg,
        "per_anomaly_type": per_type,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "per_clip.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return result
