"""Training-free anomaly scorer using frozen CLIP patch features vs a temporal reference."""
from __future__ import annotations

import numpy as np

from ..registry import SCORERS
from .base import VADScorer


@SCORERS.register("clip_vad")
class CLIPVADScorer(VADScorer):
    """Per-patch CLIP feature distance to the temporal-median reference frame.

    A photorealistic fill sits close to the reference in feature space (low
    score), so it does not penalize diffusion restoration the way a raw
    pixel-difference score does. Fully frozen; runs on GPU.
    """

    def __init__(self, model: str = "openai/clip-vit-base-patch32",
                 grid: int = 4, device: str = "cuda"):
        self.model_id = model
        self.grid = grid
        self.device = device
        self._model = None

    def _load(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = self.device if torch.cuda.is_available() else "cpu"
        self.device = dev
        self._torch = torch
        self._model = CLIPModel.from_pretrained(self.model_id).to(dev).eval()
        self._proc = CLIPProcessor.from_pretrained(self.model_id)

    def _embed(self, crops):
        from PIL import Image
        torch = self._torch
        imgs = [Image.fromarray(c) for c in crops]
        inp = self._proc(images=imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.get_image_features(**inp)
        if torch.is_tensor(out):
            f = out
        elif hasattr(out, "image_embeds"):
            f = out.image_embeds
        elif hasattr(out, "pooler_output"):
            f = out.pooler_output
        else:
            f = out[0]
        return f / f.norm(dim=-1, keepdim=True)

    def score_map(self, frames: np.ndarray) -> np.ndarray:
        if self._model is None:
            self._load()
        import cv2
        T, H, W, _ = frames.shape
        G = self.grid
        ref = np.median(frames.astype(np.float32), axis=0).astype(np.uint8)
        ys = np.linspace(0, H, G + 1).astype(int)
        xs = np.linspace(0, W, G + 1).astype(int)

        def cells(img):
            return [img[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                    for i in range(G) for j in range(G)]

        ref_emb = self._embed(cells(ref))
        heat = np.zeros((T, H, W), np.float32)
        for t in range(T):
            fe = self._embed(cells(frames[t]))
            d = (1 - (fe * ref_emb).sum(-1)).detach().cpu().numpy().reshape(G, G)
            heat[t] = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        return heat
