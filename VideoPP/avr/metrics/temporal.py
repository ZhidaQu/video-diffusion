"""Temporal-consistency metric: optical-flow warp error between frames."""
from __future__ import annotations

import numpy as np

from ..ops import warp_onto


def warp_error(frames: np.ndarray) -> float:
    import cv2
    T = frames.shape[0]
    if T < 2:
        return 0.0
    errs = []
    for t in range(T - 1):
        g0 = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(frames[t + 1], cv2.COLOR_RGB2GRAY)
        warped = warp_onto(frames[t], g0, g1)
        errs.append(np.mean(np.abs(warped.astype(np.float32)
                                   - frames[t + 1].astype(np.float32))))
    return float(np.mean(errs))
