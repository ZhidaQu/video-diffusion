"""Inpainter interface: fill masked regions of a frame sequence."""
from __future__ import annotations

import numpy as np


class Inpainter:
    def inpaint(self, frames: np.ndarray, mask: np.ndarray) -> np.ndarray:  # noqa: E501  pragma: no cover
        raise NotImplementedError
