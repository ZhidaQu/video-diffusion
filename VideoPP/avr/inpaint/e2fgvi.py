"""E2FGVI (CVPR'22) video-inpainting baseline, wrapped as an Inpainter.

Faithful in-memory port of third_party/E2FGVI/test.py. The E2FGVI model depends
on mmcv-full (custom deformable-conv CUDA op) which is incompatible with recent
torch/CUDA; run it in a SEPARATE env per BASELINES.md. Imports happen
only in _load(), so registering this does not affect the main environment.
"""
from __future__ import annotations

import os

import numpy as np

from ..registry import INPAINTERS
from .base import Inpainter


@INPAINTERS.register("e2fgvi")
class E2FGVI(Inpainter):
    def __init__(self, repo: str = "", model: str = "e2fgvi_hq", ckpt: str = "",
                 device: str = "cuda", neighbor_stride: int = 5, ref_step: int = 10):
        self.repo = repo or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "third_party", "E2FGVI")
        self.model_name = model
        self.ckpt = ckpt
        self.device = device
        self.neighbor_stride = neighbor_stride
        self.ref_step = ref_step
        self._loaded = False

    def _load(self):
        import importlib
        import sys
        import torch
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        try:
            net = importlib.import_module("model." + self.model_name)
        except Exception as e:
            raise ImportError(
                "E2FGVI needs mmcv-full in a separate env — see "
                "BASELINES.md") from e
        dev = self.device if torch.cuda.is_available() else "cpu"
        self.device = dev
        self._torch = torch
        ckpt = self.ckpt or os.path.join(self.repo, "release_model",
                                         f"{self.model_name}-CVPR22.pth")
        model = net.InpaintGenerator().to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        self.model = model.eval()
        self._loaded = True

    def _ref_index(self, neighbor_ids, length):
        return [i for i in range(0, length, self.ref_step) if i not in neighbor_ids]

    def inpaint(self, frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self._loaded:
            self._load()
        torch = self._torch
        dev = self.device
        T, H, W, _ = frames.shape

        import cv2
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        md = np.stack([cv2.dilate(mask[t].astype(np.uint8), cross, iterations=4)
                       for t in range(T)])  # match E2FGVI test.py mask dilation
        imgs = (torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0)
        imgs = (imgs.unsqueeze(0) * 2 - 1).to(dev)
        masks = torch.from_numpy(md.astype(np.float32))[None, :, None].to(dev)
        binary = [md[t][..., None] for t in range(T)]
        comp = [None] * T

        mh, mw = 60, 108
        hp, wp = (mh - H % mh) % mh, (mw - W % mw) % mw
        for f in range(0, T, self.neighbor_stride):
            nb = list(range(max(0, f - self.neighbor_stride),
                            min(T, f + self.neighbor_stride + 1)))
            rf = self._ref_index(nb, T)
            with torch.no_grad():
                mi = imgs[:1, nb + rf] * (1 - masks[:1, nb + rf])
                mi = torch.cat([mi, torch.flip(mi, [3])], 3)[:, :, :, :H + hp, :]
                mi = torch.cat([mi, torch.flip(mi, [4])], 4)[:, :, :, :, :W + wp]
                pred, _ = self.model(mi, len(nb))
                pred = ((pred[:, :, :H, :W] + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255
                for i, idx in enumerate(nb):
                    img = pred[i].astype(np.uint8) * binary[idx] + frames[idx] * (1 - binary[idx])
                    comp[idx] = (img if comp[idx] is None
                                 else (comp[idx].astype(np.float32) * 0.5
                                       + img.astype(np.float32) * 0.5)).astype(np.uint8)
        return np.stack(comp).astype(np.uint8)
