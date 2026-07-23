"""Video output. Frames are always np.uint8 [T, H, W, 3] RGB."""
from __future__ import annotations

import os

import numpy as np


def save_video(path: str, frames: np.ndarray, fps: int = 10) -> None:
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    imageio.mimsave(path, list(frames), fps=fps)


def save_triptych(path: str, anom: np.ndarray, restored: np.ndarray,
                  gt_clean: np.ndarray, fps: int = 10) -> None:
    """Side-by-side [anomalous | restored | gt-clean] qualitative video."""
    T = min(len(anom), len(restored), len(gt_clean))
    strip = np.concatenate([anom[:T], restored[:T], gt_clean[:T]], axis=2)
    save_video(path, strip, fps=fps)
