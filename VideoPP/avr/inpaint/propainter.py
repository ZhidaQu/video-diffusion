"""ProPainter (ICCV'23) video-inpainting baseline, wrapped as an Inpainter.

In-memory port of third_party/ProPainter/inference_propainter.py for short clips.
"""
from __future__ import annotations

import os

import numpy as np

from ..registry import INPAINTERS
from .base import Inpainter

_URL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"


@INPAINTERS.register("propainter")
class ProPainter(Inpainter):
    def __init__(self, repo: str = "", device: str = "cuda", raft_iter: int = 20,
                 neighbor_length: int = 10, ref_stride: int = 10,
                 mask_dilation: int = 4, fp16: bool = True):
        self.repo = repo or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "third_party", "ProPainter")
        self.device = device
        self.raft_iter = raft_iter
        self.neighbor_length = neighbor_length
        self.ref_stride = ref_stride
        self.mask_dilation = mask_dilation
        self.fp16 = fp16
        self._loaded = False

    def _load(self):
        import sys
        import torch
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        from model.modules.flow_comp_raft import RAFT_bi
        from model.propainter import InpaintGenerator
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet
        from utils.download_util import load_file_from_url

        self._torch = torch
        dev = self.device if torch.cuda.is_available() else "cpu"
        self.device = dev
        wdir = os.path.join(self.repo, "weights")
        raft = load_file_from_url(_URL + "raft-things.pth", model_dir=wdir)
        flow = load_file_from_url(_URL + "recurrent_flow_completion.pth", model_dir=wdir)
        pp = load_file_from_url(_URL + "ProPainter.pth", model_dir=wdir)
        self.raft = RAFT_bi(raft, dev)
        self.flow = RecurrentFlowCompleteNet(flow)
        for p in self.flow.parameters():
            p.requires_grad = False
        self.flow.to(dev).eval()
        self.model = InpaintGenerator(model_path=pp).to(dev).eval()
        self._loaded = True

    def _ref_index(self, f, neighbor_ids, length):
        ref = []
        for i in range(0, length, self.ref_stride):
            if i not in neighbor_ids:
                ref.append(i)
        return ref

    def inpaint(self, frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self._loaded:
            self._load()
        import cv2
        import scipy.ndimage
        torch = self._torch
        dev = self.device

        T, H, W, _ = frames.shape
        H8, W8 = H - H % 8, W - W % 8
        if (H8, W8) != (H, W):
            frames = np.stack([cv2.resize(f, (W8, H8)) for f in frames])
            mask = np.stack([cv2.resize(m.astype(np.uint8), (W8, H8),
                             interpolation=cv2.INTER_NEAREST).astype(bool) for m in mask])
        h, w = H8, W8
        ori = [f.astype(np.uint8) for f in frames]

        ft = (torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0)
        ft = (ft.unsqueeze(0) * 2 - 1).to(dev)  # [1,T,3,H,W] in [-1,1]
        md = np.stack([scipy.ndimage.binary_dilation(
            m, iterations=self.mask_dilation).astype(np.float32) for m in mask])
        mt = torch.from_numpy(md)[None, :, None].to(dev)  # [1,T,1,H,W]

        use_half = self.fp16 and dev == "cuda"
        with torch.no_grad():
            short = 12 if w <= 640 else (8 if w <= 720 else 4)
            if T > short:
                ff, fb = [], []
                for f in range(0, T, short):
                    e = min(T, f + short)
                    a, b = self.raft(ft[:, (f - 1 if f else f):e], iters=self.raft_iter)
                    ff.append(a)
                    fb.append(b)
                gt = (torch.cat(ff, dim=1), torch.cat(fb, dim=1))
            else:
                gt = self.raft(ft, iters=self.raft_iter)

            if use_half:
                ft, mt = ft.half(), mt.half()
                gt = (gt[0].half(), gt[1].half())
                self.flow.half()
                self.model.half()

            pred, _ = self.flow.forward_bidirect_flow(gt, mt)
            pred = self.flow.combine_flow(gt, pred, mt)

            masked = ft * (1 - mt)
            prop, upd = self.model.img_propagation(masked, pred, mt, "nearest")
            updated = ft * (1 - mt) + prop.view(1, T, 3, h, w) * mt
            upd = upd.view(1, T, 1, h, w)

        comp = [None] * T
        stride = self.neighbor_length // 2
        for f in range(0, T, stride):
            nb = list(range(max(0, f - stride), min(T, f + stride + 1)))
            rf = self._ref_index(f, nb, T)
            with torch.no_grad():
                pred_img = self.model(
                    updated[:, nb + rf], (pred[0][:, nb[:-1]], pred[1][:, nb[:-1]]),
                    mt[:, nb + rf], upd[:, nb + rf], len(nb))
                pred_img = ((pred_img.view(-1, 3, h, w) + 1) / 2)
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                bm = mt[0, nb].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                for i, idx in enumerate(nb):
                    img = pred_img[i].astype(np.uint8) * bm[i] + ori[idx] * (1 - bm[i])
                    comp[idx] = (img if comp[idx] is None
                                 else (comp[idx].astype(np.float32) * 0.5
                                       + img.astype(np.float32) * 0.5)).astype(np.uint8)
        return np.stack(comp).astype(np.uint8)
