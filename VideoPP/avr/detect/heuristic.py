"""Training-free background-subtraction anomaly detector."""
from __future__ import annotations

import numpy as np

from ..registry import DETECTORS
from .base import Detector


@DETECTORS.register("heuristic_bgsub")
class HeuristicBgSub(Detector):
    def __init__(self, k_sigma: float = 3.0, min_area: int = 20,
                 open_ksize: int = 3, dilate: int = 3):
        self.k_sigma = k_sigma
        self.min_area = min_area
        self.open_ksize = open_ksize
        self.dilate = dilate

    def detect(self, frames: np.ndarray) -> np.ndarray:
        import cv2

        f = frames.astype(np.float32)
        bg = np.median(f, axis=0)
        dev = np.abs(f - bg[None]).mean(axis=-1)
        thr = dev.mean() + self.k_sigma * (dev.std() + 1e-6)
        raw = dev > thr

        out = np.zeros_like(raw)
        ko = np.ones((self.open_ksize, self.open_ksize), np.uint8)
        kd = np.ones((self.dilate, self.dilate), np.uint8)
        for t in range(raw.shape[0]):
            m = raw[t].astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            keep = np.zeros_like(m)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                    keep[lab == i] = 1
            if self.dilate > 0:
                keep = cv2.dilate(keep, kd)
            out[t] = keep.astype(bool)
        return out
