"""Restoration-fidelity metrics vs GT-clean: PSNR, SSIM, (optional) LPIPS."""
from __future__ import annotations

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


def _ssim_frame(a: np.ndarray, b: np.ndarray) -> float:
    import cv2
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(a, k, 1.5)
    mu_b = cv2.GaussianBlur(b, k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a2
    sb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * sab + C2)) / (
        (mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return float(ssim_map.mean())


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over frames. a,b: [T,H,W,3] uint8."""
    return float(np.mean([_ssim_frame(a[t], b[t]) for t in range(a.shape[0])]))


class LPIPS:
    """Lazy LPIPS wrapper; returns None from __call__ when lpips is unavailable."""

    def __init__(self, net: str = "alex", device: str = "cuda"):
        self.net = net
        self.device = device
        self._model = None
        self._failed = False

    def _ensure(self):
        if self._model is not None or self._failed:
            return
        try:
            import lpips
            import torch
            self._torch = torch
            dev = self.device if torch.cuda.is_available() else "cpu"
            self.device = dev
            self._model = lpips.LPIPS(net=self.net).to(dev).eval()
        except Exception:
            self._failed = True

    def __call__(self, a: np.ndarray, b: np.ndarray):
        self._ensure()
        if self._model is None:
            return None
        torch = self._torch
        with torch.no_grad():
            def to_t(x):
                x = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 127.5 - 1.0
                return x.to(self.device)
            d = self._model(to_t(a), to_t(b))
            return float(d.mean().item())
