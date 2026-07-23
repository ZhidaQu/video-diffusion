# Training-Free Strategy: Advantages, Contribution, and Evidence

Answers: what is the training-free advantage, and what training-free method do
we propose to boost performance? Positioning grounded in a literature survey
(2023–2026); numbers are from our own runs in `runs/`.

## 1. Why training-free (the advantages we claim)

1. **No labels, no training, no per-scene tuning.** Classical VAD (FFP, MNAD,
   MemAE) trains on each dataset's normal split; a new camera means new data +
   retraining. Ours runs zero-shot on any scene. LAVAD makes the same argument:
   trained methods are "domain-specific... costly for practical deployment."
2. **Open-world anomalies.** VLM + open-vocabulary grounding localizes anomaly
   types never seen in any training set; closed-set trained detectors cannot.
3. **Upgradable by composition.** Every component is a frozen foundation model
   behind a registry interface — better backbones drop in for free.
4. **Interpretable.** The VLM names the anomaly; not a black-box reconstruction
   error.
5. **Outputs a clean video, not a score.** New capability = the task itself.

## 2. Positioning (what exists, and the gap)

- Training-free VAD — **LAVAD** (CVPR'24), **PANDA** (NeurIPS'25), **VADTree**
  (NeurIPS'25), **AnyAnomaly** (2025): they *detect / explain* anomalies and
  stop at a score or caption. No restoration, no output video.
- Detect-then-inpaint — **Grounded-SAM**, **Inpaint-Anything** (2023–24): same
  detect→segment→inpaint spine but **image-level** and **user-prompted** (a human
  names the object); no temporal handling, not anomaly-driven.
- Video inpainting — **ProPainter** (ICCV'23), **DiffuEraser** (2025),
  **MiniMax-Remover** (2025): **trained**, and the mask is **given** — they do the
  "inpaint" half, never "what to remove". **OmnimatteZero** (2025) is the nearest
  *training-free* video-removal comparator.

**Gap = our claim.** No work couples training-free anomaly *discovery* →
open-vocab detection + segmentation → temporally-consistent training-free
inpainting into one **video-output** loop. We turn training-free VAD from a
*scoring/explanation* task into an end-to-end *restoration* task, without training.

## 3. Our proposed training-free enhancement: a self-correcting loop

Our experiments show two weak points: **detection is the bottleneck** and
**per-frame diffusion flickers**. We address both with test-time, frozen-model
mechanisms — each individually established in the literature (low-risk), the
anomaly-removal composition new:

**Detect → Remove → Verify → Refine**

- **Detection boost (training-free):** prompt ensembling on the anomaly name for
  GroundingDINO (CLIP-style multi-template averaging), test-time augmentation
  (multi-scale + flip) fused with Weighted Boxes Fusion, and **SAM2 temporal mask
  propagation** to turn sparse per-frame hits into a coherent tubelet (with a
  SENTRY-style refine-before-write guard against error accumulation).
- **Temporal-consistent removal (training-free):** replace independent per-frame
  inpainting with fixed/shared noise + optical-flow-warped previous-frame
  blending — the Text2Video-Zero / FateZero / TokenFlow family of flicker
  suppressors. Implemented as `hf_diffusion(temporal=True)`.
- **Self-verification loop:** score the restored region with a frozen scorer
  (`clip_vad`); if residual anomaly persists, expand the mask and re-inpaint,
  capped at a few iterations (the literature warns self-correction saturates).
  The frozen detector acts as a test-time reward — boosting the removal metric
  with no training.

## 4. Evidence so far (from our runs)

- **Where diffusion helps (finding, `runs/*/RESULTS.md`):** on static Ped2
  backgrounds classical inpainting wins (LPIPS 0.004 vs 0.007); on dynamic
  Avenue, diffusion wins (LPIPS 0.004 vs 0.007). Diffusion pays off on richer
  scenes.
- **Temporal consistency works (verified, full Avenue benchmark, GT mask):**
  per-frame → temporal improves PSNR 43.01 → 44.52, LPIPS 0.0036 → 0.0031,
  warp_error 1.935 → 1.796 (GT clean 1.86) — consistent across all anomaly types.
  A training-free score boost.
- **Naive detection widening does NOT help (negative result):** TTA + prompt
  ensembling + lower thresholds drops mask_iou 0.033 → 0.007 on Avenue — it
  over-detects.
- **SAM2 temporal propagation DOES help (positive result):** seeding a
  GroundingDINO box then propagating with SAM2 lifts mask_iou 0.027 → 0.096
  (3.5x) on Avenue. The detection boost should be temporal propagation, not
  blind augmentation — a clean contrast that motivates the design.
- **Metric fix:** the pixel proxy scorer explodes to residual −106 on diffusion
  (per-frame flicker); the frozen CLIP scorer (`clip_vad`) keeps it in a sane
  range and gives +0.68 residual drop on moving anomalies.
- **Bottleneck confirmed:** grounded_sam mask_iou 0.07–0.40 vs GT-mask 0.82 —
  detection, not restoration, is the weak link, motivating the detection boosts.

## 5. Prior work to cite and differentiate

- Detection: LAVAD, PANDA, VADTree (detect/explain only — no output video).
- Pipeline: Grounded-SAM, Inpaint-Anything (image-level, user-prompted).
- Inpainting baselines: ProPainter, DiffuEraser (trained, mask-given);
  OmnimatteZero (nearest training-free video-removal comparator).
- Boost techniques: CLIP prompt ensembling; TTA + Weighted Boxes Fusion; SAM2
  (+ SENTRY); Text2Video-Zero / FateZero / TokenFlow / RAVE; verifier-guided
  refinement (Liao et al.; "Generate-but-Verify").

**Incremental claim (CCF-C appropriate):** first to compose training-free
anomaly discovery + prompt-ensembled/TTA/verifier-refined open-vocab detection +
SAM2 propagation → training-free temporally-consistent diffusion inpainting →
clean output video. Each block is established; the anomaly-driven, video-output
composition is unoccupied.

## 6. Remaining components to implement

- [x] Temporal-consistent diffusion (`hf_diffusion(temporal=True)`) — verified win.
- [x] Frozen CLIP scorer (`clip_vad`).
- [x] Prompt ensembling + TTA in the detector — implemented, but a negative
      result as configured (over-detection); needs verification, not blind widening.
- [x] SAM2 temporal mask propagation (`grounded_sam2`) — verified: +3.5x IoU
      over single-frame grounding, reversing the negative TTA result.
- [ ] Verifier-guided self-refinement loop (clip_vad reward, capped iterations).
- [ ] ShanghaiTech-scale runs + real-anomaly qualitative removal (download in progress).
