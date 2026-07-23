# Anomaly-Aware Video Restoration: A Training-Free Detect–Remove–Inpaint Framework


---

## 1. Problem & Motivation

Video Anomaly Detection (VAD) has matured around one output type: an **anomaly
score curve**. Frame-prediction, reconstruction, memory, and recent diffusion
methods all reduce a video to per-frame scores. But many applications
(privacy-preserving surveillance summarization, footage cleanup, data
sanitization for downstream training, content moderation) do not want a score —
they want a **cleaned video** with the anomaly removed and the underlying normal
content plausibly restored.

No existing task/benchmark targets this. We call it **Anomaly-Aware Video
Restoration (AVR)**: given a video containing anomalies, output a video in which
anomalous content is removed and replaced by temporally consistent normal
content, *without* a user-provided mask and *without* training.

**Why now / why training-free**: (i) frozen VLMs + open-vocabulary grounding
(GroundingDINO+SAM) can localize "unusual" content zero-shot; (ii) pretrained
diffusion video inpainters (DiffuEraser, ProPainter) can fill masked regions
with temporal consistency. The pieces exist; nobody has assembled or evaluated
them for anomaly removal.

---

## 2. Relation to Prior Work (and how we differ)

| Prior work | What it does | How AVR differs |
|---|---|---|
| **Bi-directional Frame Interpolation for Unsup. VAD** (WACV'23) | interpolation error → anomaly **score** | classical flow, score-only; we **output a cleaned video** and use diffusion |
| **Video Event Restoration by Keyframes / USTN-DSC** (CVPR'23) | reconstruct frames → **score** | Swin, score-only, no reinsertion; we remove-and-inpaint into an output video |
| **Diffusion-VAD** (ICIP'23, MoCoDAD, patch-diffusion'24) | diffusion reconstructs normal → **difference score** | scoring only, never outputs the restored video as the deliverable |
| **Object removal / video inpainting** (ProPainter, DiffuEraser, ROSE) | remove a **user-masked known** object | mask is **auto-produced by anomaly detection**, no human mask, anomaly-driven |
| **Reimagining Anomalies** (arXiv'24) | image-level "normal" counterfactual for **explanation** | video task, restoration deliverable, not single-image explanation |

**Novelty = the composition + reframing**: detection-driven, mask-free,
training-free, *video-output* anomaly removal, plus the first benchmark and
metric suite that make it evaluable.

---

## 3. Method

Training-free pipeline (all frozen / classical):

```
video ──▶ [Detect]  VLM names anomaly → GroundingDINO boxes → SAM masks
                    → spatio-temporal mask  M ∈ {0,1}^{T×H×W}
      ──▶ [Remove+Inpaint]  diffusion video inpainter fills M
                    → restored normal video
      ──▶ [Score check]  frozen VAD scores residual anomaly (eval only)
```

Two design axes the paper studies explicitly:
- **Temporal removal vs spatial inpainting**: dropping whole anomalous frames +
  interpolating fails for anomalies persisting across many frames; spatial
  region inpainting is more robust. We treat this as a diagnosed design choice
  (a taxonomy over anomaly duration × spatial extent), not a fixed recipe.
- **Detection is the bottleneck, not restoration**: training-free localization
  is weaker than trained VAD, so we **decouple** evaluation (GT-mask upper
  bound vs auto-mask end-to-end) to report both honestly.

---

## 4. Claims

- **C1 (primary)**: A training-free pipeline removes anomalies and produces
  temporally consistent normal content — residual anomaly drops substantially
  while fidelity to the true normal stays high.
- **C2 (supporting)**: The AVR task + three-axis metric suite (fidelity /
  residual-anomaly / temporal consistency) meaningfully discriminate restoration
  quality, and the detection vs restoration contributions are separable.
- **Anti-claims ruled out**: (a) gains are not from trivial neighbor-copy; (b)
  numbers are not inflated by hiding a weak detector (GT-mask-decoupled); (c) the
  diffusion restorer is not interchangeable with any inpainter.

---

## 5. Benchmark & Metrics

**Data via synthetic-anomaly injection** — the key that makes AVR evaluable:
inject controllable anomalies (static foreign object / moving object / local
appearance) into **normal** clips; the original clip is the **GT-clean** target
and injection yields the **GT mask**. Built from the normal splits of Ped2 /
Avenue / ShanghaiTech by `scripts/prepare_dataset.py`.

**Metrics**: PSNR / SSIM / LPIPS (+ FVD) vs GT-clean; **residual anomaly drop**
`(S(anom)−S(restored))/S(anom)` inside the GT region; optical-flow **warp
error**; **mask IoU** for the decoupled study. On real anomalies without
GT-clean: no-reference residual score + warp error + qualitative + VLM "is the
anomaly still present?".

---

## 6. Experiment Plan (summary)

| Block | Tests | Systems |
|---|---|---|
| B1 main | C1, anti-(a) | naive-copy · classical(GT) · diffusion(GT) · full-auto |
| B2 decouple | C2, anti-(b) | GT mask vs auto mask |
| B3 inpainter | anti-(c) | naive vs classical vs diffusion (GT mask) |
| B4 detector (appendix) | C2 | VLM+grounding vs CLIP-heatmap vs frozen-VAD mask |
| B5 real qualitative | C1 generality | real anomalies + failure taxonomy |

Full roadmap, run order, gates, budget: `refine-logs/EXPERIMENT_PLAN.md`.
Blocks map to the systems in `scripts/real_worker.py` and the
`experiments/exp*.sh` scripts.

---

## 7. Feasibility

- **Compute**: entirely inference; 2× L4 (24 GB) suffices at 128–256 px. Whole
  suite ≈ 15–25 GPU-h. No training.
- **Data**: datasets are small (a few GB) and only *normal* clips are needed;
  synthetic anomaly injection is pure CV.
- **Risk & mitigation**: synthetic realism → ≥3 anomaly types + real-data B5;
  weak training-free detection → decoupled reporting + "detection is future
  work"; "just gluing" critique → B3 necessity + new metrics + diagnosis carry
  the contribution.

---

## 8. Contributions

1. **New task**: Anomaly-Aware Video Restoration (mask-free, training-free,
   video-output) — a shift from scoring to restoration.
2. **Benchmark & metrics**: synthetic-injection protocol giving GT-clean pairs +
   a three-axis metric suite with detection/restoration decoupling.
3. **First training-free baseline** (VLM+grounding+SAM → diffusion inpaint) and a
   **diagnostic study** (temporal vs spatial removal; detection as bottleneck).
4. **Open, self-consistent codebase** (`avr/`) that runs end-to-end today and
   swaps in real models without structural change.
