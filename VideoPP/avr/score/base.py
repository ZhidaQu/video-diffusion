"""Anomaly scorer interface for measuring residual anomaly after removal."""
from __future__ import annotations

import numpy as np


class VADScorer:
    def score_map(self, frames: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError
