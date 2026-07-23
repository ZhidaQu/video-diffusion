# Environment Setup

## 1. Hardware

- 1 NVIDIA GPU, **≥ 12 GB VRAM** (diffusion inpainting runs at 512 px). A second
  GPU only speeds up the multi-job scripts (`run_overnight.py`, `run_real_dual.py`).
- CUDA 12.x driver. Tested on RTX A5500 / L4 (24 GB), torch 2.9 + cu128.
- ~20 GB free disk for model weights + a benchmark; datasets need more (see DATASETS.md).

## 2. Python environment

```bash
conda create -n avr python=3.10 -y
conda activate avr

# install a CUDA-matched torch first (example for CUDA 12.1); see pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

`requirements.txt` pins the runtime deps: numpy, opencv-python, imageio(-ffmpeg),
scipy, pyyaml, matplotlib, lpips, **transformers** (clip_vad scorer + GroundingDINO/SAM),
**diffusers** (Stable-Diffusion inpainting). All are required by the default pipeline.

## 3. Model weights (auto-downloaded on first use)

Nothing to download manually. On first run each frozen model is fetched **once** to a
project-local cache and reused. Paths are set in `avr/__init__.py`:

```python
HF_HOME    = <repo>/weights/hf       # transformers + diffusers
TORCH_HOME = <repo>/weights/torch    # torchvision R3D-18 (FVD)
```

To store weights elsewhere (e.g. on a large disk), export before running:

```bash
export HF_HOME=/big/disk/hf TORCH_HOME=/big/disk/torch
```

Models pulled on first use (~7 GB total): Stable-Diffusion-1.5-inpainting,
GroundingDINO-tiny, SAM-ViT-base, SAM2-hiera-large, CLIP-ViT-B/32, R3D-18 (Kinetics),
LPIPS-AlexNet. **First run needs internet.** Afterwards you can go offline:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## 4. Sanity check

After building a benchmark (see DATASETS.md):

```bash
python scripts/run_one.py --benchmark data/ped2_avr --detector gt_mask --inpainter bg_diffusion --temporal --tag smoke
```

It prints the 8 metrics and writes `runs/ped2_avr/smoke/`. If you see a metrics dict,
the environment is good.

## 5. Baselines needing a separate environment

`propainter` runs in this env directly. **E2FGVI** needs `mmcv-full` (incompatible with
torch 2.x) and must run in its own conda env — see `BASELINES.md`. Do not
install mmcv into this env.
