"""Detector that returns the ground-truth mask set via set_gt."""
from __future__ import annotations

import numpy as np

from ..ops import dilate_masks
from ..registry import DETECTORS
from .base import Detector


@DETECTORS.register("gt_mask")
class GTMaskDetector(Detector):
    def __init__(self, dilate: int = 3):
        self.dilate = dilate
        self._gt = None

    def set_gt(self, mask: np.ndarray) -> None:
        self._gt = mask

    def detect(self, frames: np.ndarray) -> np.ndarray:
        if self._gt is None:
            raise RuntimeError("GTMaskDetector.set_gt() must be called per clip.")
        return dilate_masks(self._gt, self.dilate)
