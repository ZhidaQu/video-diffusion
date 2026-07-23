"""Frechet Video Distance using a frozen Kinetics-pretrained R3D-18 backbone."""
from __future__ import annotations

import numpy as np

_MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)
_STD = np.array([0.22803, 0.22145, 0.216989], np.float32)


class FVD:
    """Set-level FVD between restored and reference clips. Returns None if the
    backbone weights are unavailable."""

    def __init__(self, device: str = "cuda", size: int = 112):
        self.device = device
        self.size = size
        self._model = None
        self._failed = False

    def _ensure(self):
        if self._model is not None or self._failed:
            return
        try:
            import torch
            from torchvision.models.video import R3D_18_Weights, r3d_18
            self._torch = torch
            dev = self.device if torch.cuda.is_available() else "cpu"
            self.device = dev
            m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            m.fc = torch.nn.Identity()
            self._model = m.to(dev).eval()
        except Exception:
            self._failed = True

    def _features(self, clips):
        import cv2
        torch = self._torch
        feats = []
        for clip in clips:
            frs = np.stack([cv2.resize(f, (self.size, self.size)) for f in clip])
            x = (frs.astype(np.float32) / 255.0 - _MEAN) / _STD
            x = torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats.append(self._model(x).cpu().numpy().reshape(-1))
        return np.stack(feats)

    @staticmethod
    def _frechet(a, b):
        from scipy.linalg import sqrtm
        mu1, mu2 = a.mean(0), b.mean(0)
        c1 = np.cov(a, rowvar=False) + 1e-6 * np.eye(a.shape[1])
        c2 = np.cov(b, rowvar=False) + 1e-6 * np.eye(b.shape[1])
        covmean = sqrtm(c1 @ c2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        diff = mu1 - mu2
        return float(diff @ diff + np.trace(c1 + c2 - 2 * covmean))

    def compute(self, restored, reference):
        self._ensure()
        if self._model is None or len(restored) < 2:
            return None
        return self._frechet(self._features(restored), self._features(reference))
