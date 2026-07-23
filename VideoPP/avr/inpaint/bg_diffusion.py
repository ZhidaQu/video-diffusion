"""Background-composited diffusion: fill holes with the temporal background revealed
in other frames, use diffusion only for pixels occluded in every frame."""
from __future__ import annotations

import numpy as np

from ..registry import INPAINTERS
from .base import Inpainter
from .hf_diffusion import HFDiffusionInpainter


@INPAINTERS.register("bg_diffusion")
class BackgroundDiffusion(Inpainter):
    def __init__(self, steps: int = 25, work_size: int = 512,
                 temporal: bool = True, blend: float = 0.5,
                 device: str = "cuda", dilate: int = 7):
        self.diff = HFDiffusionInpainter(
            steps=steps, work_size=work_size, temporal=temporal, blend=blend,
            device=device, dilate=dilate)

    def _masked_median(self, frames, mask):
        f = frames.astype(np.float32)
        f[mask] = np.nan  # ignore anomalous observations
        bg = np.nanmedian(f, axis=0)
        allnan = np.isnan(bg).any(-1)  # pixels masked in every frame
        glob = np.median(frames.astype(np.float32), axis=0)
        bg[allnan] = glob[allnan]
        return bg, allnan

    def inpaint(self, frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
        bg, always = self._masked_median(frames, mask)
        out = np.where(mask[..., None], bg[None], frames).astype(np.uint8)
        if always.any():
            resid = np.broadcast_to(always, mask.shape).copy()
            out = self.diff.inpaint(out, resid)
        return out
