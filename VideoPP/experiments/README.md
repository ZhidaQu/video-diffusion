# Experiments

One independent script per paper section. Each sources `_env.sh`, dispatches jobs
across GPUs, and writes markdown tables under `runs/<exp>/`.

## Run

```bash
# edit _env.sh first: set GPUS=(...) and uncomment the conda activate line
bash experiments/exp1_main.sh              # main comparison (Table 1 + Table 2)
bash experiments/exp2_hyperparam.sh        # hyperparameter sensitivity
bash experiments/exp3_ablation.sh          # ablation
bash experiments/exp4_temporal_consistency.sh   # selling point: temporal
bash experiments/exp5_detection_robustness.sh   # selling point: detection
```

Prerequisite: build the 60-clip benchmarks once —
`python scripts/prepare_dataset.py --dataset {ped2,avenue,shanghaitech} --raw ... --out data/<name>_night --clips 60 --T 16 --size 256`.

## Map to paper sections

| Script | Paper section | What it produces |
|---|---|---|
| `exp1_main.sh` | Main experiment | Table 1 (fixed-mask fidelity: baselines + ours), Table 2 (end-to-end systems) |
| `exp2_hyperparam.sh` | Sensitivity study | blend α, diffusion steps, detector box-threshold sweeps |
| `exp3_ablation.sh` | Ablation | remove temporal / bg-composite / SAM2 from the full method |
| `exp4_temporal_consistency.sh` | Selling point #1 | per-frame vs temporal vs bg (warp_error, tLP) + flicker figure |
| `exp5_detection_robustness.sh` | Selling point #2 | detector comparison (mask_IoU) + real-anomaly qualitative |

## Baselines (main experiment)

Baselines are **independent published methods**, not ablations of ours. They must
be **run on our benchmark** — published numbers are not reusable because no prior
work reports restoration fidelity on Ped2/Avenue/ShanghaiTech with an
anomaly-injection clean-GT protocol (inpainting papers evaluate on DAVIS/
YouTube-VOS with random masks).

Must-run, all take a given mask (feed the GT mask):
- **ProPainter** (ICCV'23) — `github.com/sczhou/ProPainter`
- **E2FGVI-HQ** (CVPR'22) — `github.com/MCG-NKU/E2FGVI`
- **DiffuEraser** (2025) — `github.com/lixiaowen-xw/DiffuEraser`
- **OmnimatteZero** (SIGGRAPH Asia'25, training-free peer) — `github.com/dvirsamuel/OmnimatteZero`

To add one: clone it, write an adapter under `avr/inpaint/` that registers an
`INPAINTERS` name and implements `inpaint(frames[T,H,W,3], mask[T,H,W]) -> frames`,
then add a line to `exp1_main.sh` Table 1.

## Metric placement (convention)

- **Table 1 (fixed mask)**: PSNR, SSIM, LPIPS, FVD, warp_error, tLP, residual_drop.
  Not mask_IoU — every method shares the same mask.
- **mask_IoU** is a detection-quality metric → reported in `exp5` (detector table).
