"""Per-frame diffusion inpainting via HuggingFace diffusers."""
from __future__ import annotations

import numpy as np

from ..ops import warp_onto
from ..registry import INPAINTERS
from .base import Inpainter


def _round8(x: int) -> int:
    return max(8, int(round(x / 8)) * 8)


@INPAINTERS.register("hf_diffusion")
class HFDiffusionInpainter(Inpainter):
    def __init__(self,
                 model: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
                 prompt: str = "clean empty background, seamless, photorealistic",
                 negative_prompt: str = "object, person, text, artifact, blur",
                 steps: int = 25, guidance: float = 7.5, work_size: int = 512,
                 dilate: int = 7, device: str = "cuda", seed: int = 0,
                 temporal: bool = False, blend: float = 0.5):
        self.model = model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.guidance = guidance
        self.work_size = work_size
        self.dilate = dilate
        self.device = device
        self.seed = seed
        self.temporal = temporal
        self.blend = blend
        self._pipe = None

    def _load(self):
        import torch
        from diffusers import AutoPipelineForInpainting
        dev = self.device if torch.cuda.is_available() else "cpu"
        self.device = dev
        self._torch = torch
        dtype = torch.float16 if dev == "cuda" else torch.float32
        self._pipe = AutoPipelineForInpainting.from_pretrained(
            self.model, torch_dtype=dtype, safety_checker=None).to(dev)
        self._pipe.set_progress_bar_config(disable=True)

    def inpaint(self, frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            self._load()
        import cv2
        from PIL import Image
        torch = self._torch

        T, H, W, _ = frames.shape
        s = self.work_size
        Wr, Hr = _round8(min(s, W)), _round8(min(s, H))
        kd = np.ones((self.dilate, self.dilate), np.uint8)
        out = frames.copy()
        prev = None
        for t in range(T):
            if not mask[t].any():
                out[t] = frames[t]
                prev = out[t]
                continue
            m = cv2.dilate(mask[t].astype(np.uint8) * 255, kd)
            img = cv2.resize(frames[t], (Wr, Hr), interpolation=cv2.INTER_AREA)
            mm = cv2.resize(m, (Wr, Hr), interpolation=cv2.INTER_NEAREST)
            seed = self.seed if self.temporal else self.seed + t
            gen = torch.Generator(device=self.device).manual_seed(seed)
            res = self._pipe(prompt=self.prompt,
                             negative_prompt=self.negative_prompt,
                             image=Image.fromarray(img),
                             mask_image=Image.fromarray(mm),
                             num_inference_steps=self.steps,
                             guidance_scale=self.guidance,
                             generator=gen).images[0]
            res = cv2.resize(np.asarray(res), (W, H), interpolation=cv2.INTER_CUBIC)
            frame_out = np.where(mask[t][..., None], res, frames[t]).astype(np.uint8)
            if self.temporal and prev is not None:
                frame_out = self._blend_prev(frame_out, prev, frames[t], mask[t])
            out[t] = frame_out
            prev = out[t]
        return out

    def _blend_prev(self, cur, prev, orig, mask2d):
        """Blend the inpainted region with the flow-warped previous restored frame."""
        import cv2
        g0 = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        gp = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        warped = warp_onto(prev, gp, g0)
        blended = (self.blend * cur.astype(np.float32)
                   + (1 - self.blend) * warped.astype(np.float32))
        return np.where(mask2d[..., None], blended, cur).astype(np.uint8)
