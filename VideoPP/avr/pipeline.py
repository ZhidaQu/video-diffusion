"""AVR pipeline: detect -> mask -> inpaint -> clean video."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .detect.base import Detector
from .inpaint.base import Inpainter


class AVRPipeline:
    def __init__(self, detector: Detector, inpainter: Inpainter):
        self.detector = detector
        self.inpainter = inpainter

    def run(self, frames: np.ndarray, gt_mask: Optional[np.ndarray] = None) -> dict:
        if gt_mask is not None and hasattr(self.detector, "set_gt"):
            self.detector.set_gt(gt_mask)
        mask = self.detector.detect(frames)
        restored = self.inpainter.inpaint(frames, mask)
        return {"mask": mask, "restored": restored}
