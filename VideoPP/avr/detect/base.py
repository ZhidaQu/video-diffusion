"""Detector interface: frames uint8 [T,H,W,3] -> anomaly mask bool [T,H,W]."""
from __future__ import annotations

import numpy as np


class Detector:
    def detect(self, frames: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError
