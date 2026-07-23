# Evaluation Protocol

Complete protocol for Anomaly-Aware Video Restoration (AVR). All metrics are
computed by `avr/runner.py:run_spec`; tables are produced by `scripts/run_all.py`,
`run_real_dual.py`, `run_overnight.py`, and collated by `collate.py`.

## 1. Datasets and splits

| Dataset | Use | Split |
|---|---|---|
| UCSD Ped2 | clean source for injection | Train (normal-only) |
| CUHK Avenue | clean source for injection | training_videos (normal-only) |
| ShanghaiTech | clean source for injection | SHANGHAI_TRAIN (normal-only) |
| ShanghaiTech | real-anomaly qualitative | SHANGHAI_Test (frame-level labels) |

## 2. Benchmark construction

**Quantitative (has ground truth).** From each dataset's *normal* clips we inject
a controllable anomaly and keep the triple `(anomalous, clean, mask)`:
the original normal clip is the restoration target `clean`, and the injection
region is the ground-truth `mask`. Three anomaly types are used so results are
not an artifact of one setting: `static_object`, `moving_object`,
`local_appearance`. Standard configuration: `T=16` frames, `256px`, `N` clips
(60 for the main tables). Built by `scripts/prepare_dataset.py`.

**Qualitative (real anomalies).** ShanghaiTech test clips contain real anomalous
events with frame-level labels only (no pixel masks). The anomaly is localized
open-world by `grounded_sam2`; there is no `clean` reference, so only
no-reference metrics and visual inspection apply. Built by `scripts/case_real.py`.

## 3. Systems compared

- **Detection**: `gt_mask` (GT mask — restoration upper bound), `grounded_sam`
  (GroundingDINO+SAM, single-frame), `grounded_sam2` (GroundingDINO seed +
  SAM2 temporal propagation), `heuristic_bgsub` (classical background subtraction).
- **Inpainting**: `copy_bg`, `spatial_cv2`, `flow_fill` (classical baselines);
  `hf_diffusion` (per-frame diffusion) and its `temporal=True` variant;
  `bg_diffusion` (background-composited: temporal background where revealed,
  diffusion for persistently-occluded pixels).

## 4. Metrics

| Group | Metric | Dir | Definition |
|---|---|---|---|
| Fidelity vs clean | PSNR | ↑ | pixel PSNR |
| | SSIM | ↑ | windowed SSIM |
| | LPIPS | ↓ | AlexNet perceptual distance |
| | FVD | ↓ | Fréchet distance of R3D-18 (Kinetics) video features |
| Temporal | warp_error | ↓ | optical-flow warp residual between frames |
| | tLP | ↓ | LPIPS between consecutive restored frames (flicker) |
| Removal | residual_drop | ↑ | `(S(anom)-S(restored))/S(anom)` in region, frozen CLIP scorer |
| Localization | mask_iou | ↑ | auto mask vs GT mask (temporal mean) |

Fidelity + FVD measure how close the restored video is to the true normal;
warp_error + tLP measure temporal stability; residual_drop measures whether the
anomaly is actually gone; mask_iou measures localization quality.

## 5. Evaluation design

- **Decoupled detection vs restoration**: run every inpainter with the `gt_mask`
  mask (restoration upper bound) and with an automatic detector (end-to-end).
  The gap isolates detector error from restoration quality.
- **Per-anomaly-type breakdown**: metrics are also reported per injected type.
- **Ablations**: temporal on/off; blend `α ∈ {0.3, 0.5, 0.7}`; detector
  single-frame (`grounded_sam`) vs temporal propagation (`grounded_sam2`).

## 6. Caveats (reported honestly)

- **FVD** needs many samples for a stable covariance; at `N=60` the covariance is
  regularized (`+εI`) and FVD should be read as a relative indicator, not an
  absolute score.
- **residual_drop** uses a single-clip training-free scorer whose temporal
  reference is polluted by persistent anomalies, so `static_object` /
  `local_appearance` are scored pessimistically; `moving_object` is scored fairly.
- **Real anomalies** have no clean reference — only warp_error, tLP,
  residual_drop and visual inspection apply.

## 7. Reproduce

```bash
python scripts/prepare_dataset.py --dataset ped2 --raw data/raw          # build benchmark
python scripts/run_all.py --benchmark data/ped2_avr                      # classical + baselines
python scripts/run_real_dual.py --benchmark data/ped2_real --systems gt_diffusion,gt_diffusion_temporal
python scripts/run_overnight.py --benchmarks data/ped2_night,data/avenue_night,data/sht_night
python scripts/collate.py data/ped2_night data/avenue_night data/sht_night
```
