"""Training-free detector: GroundingDINO seed box -> SAM2 temporal mask propagation."""
from __future__ import annotations

import numpy as np

from ..ops import dilate_masks
from ..registry import DETECTORS
from .base import Detector


@DETECTORS.register("grounded_sam2")
class GroundedSAM2(Detector):
    def __init__(self,
                 prompt: str = "an unusual object. an anomaly. a foreign object.",
                 grounding_model: str = "IDEA-Research/grounding-dino-tiny",
                 sam2_model: str = "facebook/sam2-hiera-large",
                 box_threshold: float = 0.25, text_threshold: float = 0.20,
                 device: str = "cuda", dilate: int = 5, seed_frames: int = 6):
        self.prompt = prompt if prompt.strip().endswith(".") else prompt + "."
        self.grounding_model = grounding_model
        self.sam2_model = sam2_model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.dilate = dilate
        self.seed_frames = seed_frames
        self._loaded = False

    def _load(self):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor, Sam2VideoModel, Sam2VideoProcessor)
        dev = self.device if torch.cuda.is_available() else "cpu"
        self.device = dev
        self._torch = torch
        self.gd_proc = AutoProcessor.from_pretrained(self.grounding_model)
        self.gd = (AutoModelForZeroShotObjectDetection
                   .from_pretrained(self.grounding_model).to(dev).eval())
        self.sam2 = Sam2VideoModel.from_pretrained(self.sam2_model).to(dev).eval()
        self.sam2_proc = Sam2VideoProcessor.from_pretrained(self.sam2_model)
        self._loaded = True

    def _gd_box(self, frame):
        """Union bounding box of all GroundingDINO detections, or None."""
        from PIL import Image
        torch = self._torch
        H, W = frame.shape[:2]
        inp = self.gd_proc(images=Image.fromarray(frame), text=self.prompt.lower(),
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.gd(**inp)
        res = self.gd_proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[(H, W)])[0]
        b = res["boxes"]
        if b.shape[0] == 0:
            return None
        x0 = float(b[:, 0].min()); y0 = float(b[:, 1].min())
        x1 = float(b[:, 2].max()); y1 = float(b[:, 3].max())
        return [x0, y0, x1, y1]

    def detect(self, frames: np.ndarray) -> np.ndarray:
        if not self._loaded:
            self._load()
        torch = self._torch
        T, H, W, _ = frames.shape

        seed_idx, box = None, None
        for t in range(min(self.seed_frames, T)):
            box = self._gd_box(frames[t])
            if box is not None:
                seed_idx = t
                break
        out = np.zeros((T, H, W), dtype=bool)
        if seed_idx is None:
            return out

        sess = self.sam2_proc.init_video_session(
            video=[frames[t] for t in range(T)], inference_device=self.device,
            dtype=torch.float32)
        self.sam2_proc.add_inputs_to_inference_session(
            inference_session=sess, frame_idx=seed_idx, obj_ids=1,
            input_boxes=[[box]])

        def collect(reverse):
            for o in self.sam2.propagate_in_video_iterator(
                    sess, start_frame_idx=seed_idx, reverse=reverse):
                m = self.sam2_proc.post_process_masks(
                    [o.pred_masks], original_sizes=[(H, W)], binarize=True)[0]
                out[o.frame_idx] = np.asarray(m.cpu()).reshape(-1, H, W).any(axis=0)

        collect(False)
        if seed_idx > 0:
            collect(True)
        return dilate_masks(out, self.dilate)
