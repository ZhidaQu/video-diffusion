# External baselines for the main experiment

Independent published video-inpainting baselines. Each is fed the SAME mask as
our method (feed the GT mask) so the comparison isolates restoration quality.

Every adapter implements the same contract as our own inpainters:
`inpaint(frames[T,H,W,3] uint8, mask[T,H,W] bool) -> frames[T,H,W,3] uint8`,
registered under an `INPAINTERS` name, imported lazily so it never affects the
main environment.

## Status

| Baseline | Registry name | Runs in this env? | Adapter |
|---|---|---|---|
| ProPainter (ICCV'23) | `propainter` | **Yes** (pure PyTorch) | `avr/inpaint/propainter.py` — working, tested |
| E2FGVI-HQ (CVPR'22) | `e2fgvi` | No (needs mmcv-full) | `avr/inpaint/e2fgvi.py` — written, run in a separate env |
| DiffuEraser (2025) | `diffueraser` | Maybe (heavy diffusion) | integration recipe below |
| OmnimatteZero (SIGGRAPH Asia'25) | `omnimattezero` | No (needs a video-diffusion backbone) | integration recipe below |

## ProPainter — ready to use

Already cloned to `third_party/ProPainter`; weights auto-download to its
`weights/` on first run. Just use `--inpainter propainter` (already wired into
`experiments/exp1_main.sh`). Nothing else to do.

## E2FGVI — needs a separate env (do NOT install into the main env)

The E2FGVI model uses `mmcv-full` (a custom deformable-conv CUDA op) plus
`mmcv.runner`, which are incompatible with torch 2.x / recent CUDA. Set up an
isolated env, then run only the E2FGVI jobs from there.

```bash
conda create -n e2fgvi python=3.8 -y && conda activate e2fgvi
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install scikit-image tqdm  # + this repo's runtime deps (numpy, opencv, scipy, imageio, lpips, transformers, diffusers, pyyaml)
# weights: download E2FGVI-HQ-CVPR22.pth (Google Drive, see third_party/E2FGVI/README.md)
#          into third_party/E2FGVI/release_model/
```

The adapter (`avr/inpaint/e2fgvi.py`) is a faithful port of `E2FGVI/test.py`;
once the env + weights are in place, `--inpainter e2fgvi` works unchanged.

## DiffuEraser (2025) — integration recipe

Repo: `github.com/lixiaowen-xw/DiffuEraser`. Diffusion (SD + BrushNet + ProPainter
prior). Heavy VRAM. Clone into `third_party/DiffuEraser`, download its weights,
then add `avr/inpaint/diffueraser.py` following the ProPainter adapter pattern:

1. `_load()`: add the repo to `sys.path`, import its pipeline (e.g.
   `from diffueraser.diffueraser import DiffuEraser`), build it once.
2. `inpaint(frames, mask)`: write frames+masks to a temp dir (DiffuEraser's API
   is video/dir-based), call its inference, read the output frames back, return
   `uint8 [T,H,W,3]`. Register `@INPAINTERS.register("diffueraser")`.
3. Add a queue line to `exp1_main.sh` Table 1: `--inpainter diffueraser`.

## OmnimatteZero (2025, training-free peer) — integration recipe

Repo: `github.com/dvirsamuel/OmnimatteZero`. Training-free removal on a pretrained
video-diffusion backbone; run in its own env (backbone deps differ from ours).
Same adapter pattern: import its removal function in `_load()`, wrap it in
`inpaint(frames, mask)`, register `@INPAINTERS.register("omnimattezero")`, add to
`exp1_main.sh`. This is the key training-free-vs-training-free comparison — cite
and run it if time allows.

## Note on reusing numbers

None of these publish restoration-fidelity numbers on Ped2/Avenue/ShanghaiTech
(they evaluate on DAVIS/YouTube-VOS with random masks), so their numbers are not
reusable — each must be run here on our anomaly-injection benchmark.
