#!/usr/bin/env bash
# EXP5 - Selling point #2: detection robustness. Same inpainter (bg_diffusion +
# temporal); vary only the training-free detector. SAM2 temporal propagation
# should beat single-frame grounding and the classical detector. Read mask_iou
# (localization) and the end-to-end fidelity (lpips/psnr).
source "$(dirname "$0")/_env.sh"

for B in data/avenue_night data/sht_night; do
  echo "==== EXP5 detection on $B ===="
  queue \
   "--benchmark $B --detector heuristic_bgsub --inpainter bg_diffusion --temporal --tag e5_heuristic" \
   "--benchmark $B --detector grounded_sam  --box-threshold 0.25 --text-threshold 0.20 --dilate 5 --inpainter bg_diffusion --temporal --tag e5_grounded_sam" \
   "--benchmark $B --detector grounded_sam2 --box-threshold 0.25 --text-threshold 0.20 --dilate 5 --inpainter bg_diffusion --temporal --tag e5_grounded_sam2"
  python scripts/table.py --benchmark "$B" \
   --tags e5_heuristic,e5_grounded_sam,e5_grounded_sam2 \
   --out "runs/exp5_detection/$(basename "$B").md"
done

# Qualitative real-anomaly removal (open-world detection on ShanghaiTech test).
CUDA_VISIBLE_DEVICES=${GPUS[0]} python scripts/case_real.py --video 02_0161 \
  --prompt "a bicycle. a person riding a bicycle." --out runs/exp5_detection/real_bicycle || true
