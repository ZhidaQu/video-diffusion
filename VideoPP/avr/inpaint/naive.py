"""Training-free classical inpainting baselines."""
from __future__ import annotations

import numpy as np

from ..ops import warp_onto
from ..registry import INPAINTERS
from .base import Inpainter


@INPAINTERS.register("copy_bg")
class CopyBackground(Inpainter):
    """Replace masked pixels with the per-pixel temporal median background."""

    def inpaint(self, frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
        f = frames.astype(np.float32)
        bg = np.median(f, axis=0)
        out = f.copy()
        m = mask[..., None]
        out = np.where(m, bg[None], out)
        return np.clip(out, 0, 255).astype(np.uint8)


@INPAINTERS.register("spatial_cv2")
class SpatialCV2(Inpainter):
    """Per-frame classical spatial inpainting (Telea or Navier-Stokes)."""

    def __init__(self, method: str = "telea", radius: int = 5):
        self.method = method
        self.radius = radius

    def inpaint(self, frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import cv2
        flag = cv2.INPAINT_TELEA if self.method == "telea" else cv2.INPAINT_NS
        out = np.empty_like(frames)
        for t in range(frames.shape[0]):
            m = (mask[t].astype(np.uint8)) * 255
            bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
            res = cv2.inpaint(bgr, m, self.radius, flag)
            out[t] = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        return out


@INPAINTERS.register("flow_fill")
class FlowFill(Inpainter):
    """Fill each hole by warping the previous filled frame via Farneback flow, with Telea fallback."""

    def __init__(self, radius: int = 5):
        self.radius = radius

    def inpaint(self, frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import cv2
        T = frames.shape[0]
        out = frames.copy()
        m0 = (mask[0].astype(np.uint8)) * 255
        bgr = cv2.cvtColor(out[0], cv2.COLOR_RGB2BGR)
        out[0] = cv2.cvtColor(cv2.inpaint(bgr, m0, self.radius, cv2.INPAINT_TELEA),
                              cv2.COLOR_BGR2RGB)
        for t in range(1, T):
            prev_g = cv2.cvtColor(out[t - 1], cv2.COLOR_RGB2GRAY)
            cur_g = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
            warped = warp_onto(out[t - 1], prev_g, cur_g)
            m = mask[t][..., None]
            filled = np.where(m, warped, frames[t])
            mm = (mask[t].astype(np.uint8)) * 255
            bgr = cv2.cvtColor(filled.astype(np.uint8), cv2.COLOR_RGB2BGR)
            filled = cv2.cvtColor(
                cv2.inpaint(bgr, cv2.erode(mm, np.ones((3, 3), np.uint8)),
                            self.radius, cv2.INPAINT_TELEA),
                cv2.COLOR_BGR2RGB)
            out[t] = filled
        return out
