"""Shared image ops: optical-flow warping and mask dilation."""
from __future__ import annotations

import numpy as np


def warp_onto(src: np.ndarray, src_gray: np.ndarray, dst_gray: np.ndarray) -> np.ndarray:
    """Warp `src` onto `dst`'s pixel grid via Farneback flow (dst -> src)."""
    import cv2
    flow = cv2.calcOpticalFlowFarneback(dst_gray, src_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = dst_gray.shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mapx = (gx + flow[..., 0]).astype(np.float32)
    mapy = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def dilate_masks(masks: np.ndarray, ksize: int) -> np.ndarray:
    """Dilate each frame of a [T,H,W] bool mask with a ksize x ksize kernel."""
    import cv2
    if ksize <= 0:
        return masks
    k = np.ones((ksize, ksize), np.uint8)
    return np.stack([cv2.dilate(m.astype(np.uint8), k).astype(bool)
                     for m in masks], axis=0)
