"""Training-free detector: (optional VLM) -> GroundingDINO -> SAM, with optional
prompt ensembling and flip test-time augmentation."""
from __future__ import annotations

import numpy as np

from ..ops import dilate_masks
from ..registry import DETECTORS
from .base import Detector


@DETECTORS.register("grounded_sam")
class GroundedSAM(Detector):
    def __init__(self,
                 prompt: str = "an unusual object. an anomaly. a foreign object.",
                 grounding_model: str = "IDEA-Research/grounding-dino-tiny",
                 sam_model: str = "facebook/sam-vit-base",
                 box_threshold: float = 0.25, text_threshold: float = 0.20,
                 device: str = "cuda", dilate: int = 5,
                 vlm: str = "", sample_stride: int = 1,
                 prompts=None, tta: bool = False):
        self.prompt = prompt if prompt.strip().endswith(".") else prompt + "."
        self.grounding_model = grounding_model
        self.sam_model = sam_model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.dilate = dilate
        self.vlm = vlm
        self.sample_stride = max(1, sample_stride)
        self.prompts = prompts
        self.tta = tta
        self._loaded = False

    def _load(self):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor, SamModel, SamProcessor)
        dev = self.device if torch.cuda.is_available() else "cpu"
        self.device = dev
        self._torch = torch
        self.gd_proc = AutoProcessor.from_pretrained(self.grounding_model)
        self.gd = (AutoModelForZeroShotObjectDetection
                   .from_pretrained(self.grounding_model).to(dev).eval())
        self.sam = SamModel.from_pretrained(self.sam_model).to(dev).eval()
        self.sam_proc = SamProcessor.from_pretrained(self.sam_model)
        self._vlm_pipe = None
        if self.vlm:
            from transformers import pipeline
            self._vlm_pipe = pipeline("image-to-text", model=self.vlm,
                                      device=0 if dev == "cuda" else -1)
        self._loaded = True

    def _vlm_prompt(self, frame: np.ndarray) -> str:
        if self._vlm_pipe is None:
            return self.prompt
        from PIL import Image
        try:
            cap = self._vlm_pipe(Image.fromarray(frame))[0]["generated_text"]
            cap = cap.strip().rstrip(".")
            return f"{cap}. {self.prompt}" if cap else self.prompt
        except Exception:
            return self.prompt

    def _gd_boxes(self, img, prompt, H, W):
        torch = self._torch
        inp = self.gd_proc(images=img, text=prompt.lower(),
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.gd(**inp)
        res = self.gd_proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[(H, W)])[0]
        return res["boxes"]

    def _sam_union(self, img, boxes):
        torch = self._torch
        sam_in = self.sam_proc(img, input_boxes=[[b.tolist() for b in boxes]],
                               return_tensors="pt").to(self.device)
        with torch.no_grad():
            sam_out = self.sam(**sam_in)
        masks = self.sam_proc.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(), sam_in["original_sizes"].cpu(),
            sam_in["reshaped_input_sizes"].cpu())[0]
        m = np.asarray(masks)
        if m.ndim == 4:
            m = m[:, 0]
        return m.any(axis=0).astype(bool)

    def _detect_frame(self, frame: np.ndarray, prompt: str) -> np.ndarray:
        from PIL import Image
        torch = self._torch
        H, W = frame.shape[:2]
        img = Image.fromarray(frame)
        prompts = self.prompts or [prompt]
        collected = []
        for p in prompts:
            b = self._gd_boxes(img, p, H, W)
            if b.shape[0]:
                collected.append(b)
            if self.tta:
                bf = self._gd_boxes(Image.fromarray(frame[:, ::-1].copy()), p, H, W)
                if bf.shape[0]:
                    flipped = bf.clone()
                    flipped[:, 0] = W - bf[:, 2]
                    flipped[:, 2] = W - bf[:, 0]
                    collected.append(flipped)
        if not collected:
            return np.zeros((H, W), bool)
        return self._sam_union(img, torch.cat(collected, dim=0))

    def detect(self, frames: np.ndarray) -> np.ndarray:
        if not self._loaded:
            self._load()
        T, H, W, _ = frames.shape
        out = np.zeros((T, H, W), dtype=bool)
        last = np.zeros((H, W), dtype=bool)
        for t in range(T):
            if t % self.sample_stride == 0:
                last = self._detect_frame(frames[t], self._vlm_prompt(frames[t]))
            out[t] = last
        return dilate_masks(out, self.dilate)
