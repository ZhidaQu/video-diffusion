"""Inject synthetic anomalies into a normal clip, returning (anom, clean, mask)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..registry import ANOMALIES

Triple = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (anom, clean, mask)


class AnomalyInjector:
    """Base class. Subclasses implement `inject`."""

    def inject(self, clean: np.ndarray, seed: int = 0) -> Triple:
        raise NotImplementedError


def _blob_mask(H: int, W: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


@ANOMALIES.register("static_object")
@dataclass
class StaticObject(AnomalyInjector):
    """A foreign colored object appears at a fixed location for the whole clip."""
    radius: int = 12
    color: Tuple[int, int, int] = (220, 30, 30)

    def inject(self, clean: np.ndarray, seed: int = 0) -> Triple:
        rng = np.random.default_rng(seed)
        T, H, W, _ = clean.shape
        cy = int(rng.integers(self.radius, H - self.radius))
        cx = int(rng.integers(self.radius, W - self.radius))
        m2d = _blob_mask(H, W, cy, cx, self.radius)
        anom = clean.copy()
        mask = np.zeros((T, H, W), dtype=bool)
        for t in range(T):
            anom[t][m2d] = np.array(self.color, dtype=np.uint8)
            mask[t] = m2d
        return anom, clean, mask


@ANOMALIES.register("moving_object")
@dataclass
class MovingObject(AnomalyInjector):
    """A foreign object translates across the scene (background-sub friendly)."""
    radius: int = 10
    color: Tuple[int, int, int] = (30, 220, 60)

    def inject(self, clean: np.ndarray, seed: int = 0) -> Triple:
        rng = np.random.default_rng(seed)
        T, H, W, _ = clean.shape
        y0 = int(rng.integers(self.radius, H - self.radius))
        x0 = self.radius
        x1 = W - self.radius
        anom = clean.copy()
        mask = np.zeros((T, H, W), dtype=bool)
        for t in range(T):
            frac = t / max(T - 1, 1)
            cx = int(x0 + frac * (x1 - x0))
            cy = int(y0 + 8 * np.sin(2 * np.pi * frac))
            cy = int(np.clip(cy, self.radius, H - self.radius))
            m2d = _blob_mask(H, W, cy, cx, self.radius)
            anom[t][m2d] = np.array(self.color, dtype=np.uint8)
            mask[t] = m2d
        return anom, clean, mask


@ANOMALIES.register("local_appearance")
@dataclass
class LocalAppearance(AnomalyInjector):
    """A region undergoes an abnormal appearance change (over-bright flicker)."""
    radius: int = 16
    gain: float = 1.8

    def inject(self, clean: np.ndarray, seed: int = 0) -> Triple:
        rng = np.random.default_rng(seed)
        T, H, W, _ = clean.shape
        cy = int(rng.integers(self.radius, H - self.radius))
        cx = int(rng.integers(self.radius, W - self.radius))
        m2d = _blob_mask(H, W, cy, cx, self.radius)
        anom = clean.copy().astype(np.float32)
        mask = np.zeros((T, H, W), dtype=bool)
        for t in range(T):
            flick = 1.0 + (self.gain - 1.0) * (0.5 + 0.5 * np.sin(t))
            anom[t][m2d] = np.clip(anom[t][m2d] * flick, 0, 255)
            mask[t] = m2d
        return anom.astype(np.uint8), clean, mask
