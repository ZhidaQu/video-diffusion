# AVR — Anomaly-Aware Video Restoration (training-free)

Detect an anomaly in a video → build a spatio-temporal mask → remove & inpaint →
output a **clean** video. Fully **training-free**: every component is a frozen
pretrained model or a classical algorithm.

- **Start here** — environment setup: [`ENVIRONMENT.md`](ENVIRONMENT.md) · datasets: [`DATASETS.md`](DATASETS.md)
- Research framing, contribution, evidence: [`STRATEGY.md`](STRATEGY.md), [`PROPOSAL.md`](PROPOSAL.md)
- Evaluation protocol and metrics: [`EVALUATION.md`](EVALUATION.md)
- Per-paper-section experiment scripts: [`experiments/`](experiments/)
- External baselines (ProPainter, E2FGVI, …): [`BASELINES.md`](BASELINES.md)

## Components

| Stage | Registry names |
|-------|----------------|
| Detect / localize | `gt_mask` (GT mask, upper bound), `heuristic_bgsub`, `grounded_sam`, `grounded_sam2` |
| Remove / inpaint | `copy_bg`, `spatial_cv2`, `flow_fill`, `hf_diffusion` (`temporal=True`), `bg_diffusion`, `propainter`, `e2fgvi` |
| Residual-anomaly scorer | `clip_vad` |

Pretrained weights download once to a project-local `weights/` on `/data` (set via
`HF_HOME`/`TORCH_HOME` in `avr/__init__.py`), never the home dir. Components are
imported lazily, so registering one never affects the environment.

## Quickstart

```bash
pip install -r requirements.txt

# 1) build a benchmark: inject anomalies into a dataset's normal clips
python scripts/prepare_dataset.py --dataset ped2 --raw data/raw --out data/ped2_avr --clips 60 --T 16 --size 256
#   (datasets: ped2 / avenue / shanghaitech; see EVALUATION.md for sources)

# 2) run one system
python scripts/run_one.py --benchmark data/ped2_avr --detector gt_mask --inpainter bg_diffusion --temporal --tag ours

# 3) run a paper section end-to-end (edit experiments/_env.sh first: GPUS + conda env)
bash experiments/exp1_main.sh          # main comparison
bash experiments/exp3_ablation.sh      # ablation
```

Each run writes `runs/<benchmark>/<tag>/results.json`, `per_clip.csv`, and a few
`samples/*.mp4` triptychs `[anomalous | restored | gt-clean]`.

## Metrics (8, computed by `avr/runner.py`)

- Fidelity vs GT-clean: **PSNR, SSIM, LPIPS, FVD**
- Temporal consistency: **warp_error, tLP**
- Anomaly removal: **residual_drop** (frozen CLIP scorer)
- Localization: **mask_iou** (auto mask vs GT)

## Layout

```
avr/        data/ · detect/ · inpaint/ · score/ · metrics/ · pipeline · runner · registry
scripts/    prepare_dataset · run_one · run_all · run_real_dual · run_overnight · collate · table · case_*
experiments/  exp1_main … exp5, _env.sh, README.md  (one script per paper section)
third_party/  ProPainter, E2FGVI (baselines), BASELINES.md
```
