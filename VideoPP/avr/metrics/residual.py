"""Residual-anomaly metric: anomaly-score drop inside the GT mask after removal."""
from __future__ import annotations

import numpy as np

from ..score.base import VADScorer


def _masked_mean(heat: np.ndarray, mask: np.ndarray) -> float:
    if mask.any():
        return float(heat[mask].mean())
    return float(heat.mean())


def residual_anomaly_drop(scorer: VADScorer, anom: np.ndarray,
                          restored: np.ndarray, mask: np.ndarray,
                          eps: float = 1e-6) -> dict:
    heat_a = scorer.score_map(anom)
    heat_r = scorer.score_map(restored)
    s_a = _masked_mean(heat_a, mask)
    s_r = _masked_mean(heat_r, mask)
    drop = (s_a - s_r) / (s_a + eps)
    return {
        "score_anom": s_a,
        "score_restored": s_r,
        "residual_drop": float(drop),
    }
