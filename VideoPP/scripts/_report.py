"""Shared reporting helpers for the run/collate/table scripts."""
import numpy as np

METRICS = ["psnr", "ssim", "lpips", "tlp", "fvd", "warp_error",
           "residual_drop", "mask_iou"]


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def row(name, agg):
    return "| " + name + " | " + " | ".join(fmt(agg.get(m)) for m in METRICS) + " |"


def overlay(frame, mask):
    """Tint the masked region red on a copy of the frame."""
    red = np.zeros_like(frame)
    red[..., 0] = 255
    return np.where(mask[..., None], (0.5 * frame + 0.5 * red).astype(np.uint8), frame)
