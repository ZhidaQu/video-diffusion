#!/usr/bin/env bash
# EXP4 - Selling point #1: training-free temporal consistency. Naive per-frame
# diffusion flickers (and can lose to classical); our fixed-noise + flow-warp
# temporal mechanism and background compositing fix it. Read warp_error and tLP.
source "$(dirname "$0")/_env.sh"

for B in data/ped2_night data/avenue_night data/sht_night; do
  echo "==== EXP4 temporal on $B ===="
  queue \
   "--benchmark $B --detector gt_mask --inpainter hf_diffusion            --tag e4_perframe" \
   "--benchmark $B --detector gt_mask --inpainter hf_diffusion --temporal --tag e4_temporal" \
   "--benchmark $B --detector gt_mask --inpainter bg_diffusion --temporal --tag e4_bg_temporal"
  python scripts/table.py --benchmark "$B" \
   --tags e4_perframe,e4_temporal,e4_bg_temporal \
   --out "runs/exp4_temporal/$(basename "$B").md"
done

# Qualitative: per-frame error / flicker curves (per-frame diffusion spikes).
CUDA_VISIBLE_DEVICES=${GPUS[0]} python scripts/case_diff.py \
  --benchmark data/avenue_night --anomaly-type moving_object \
  --systems gt_flow,gt_diffusion,gt_diffusion_temporal \
  --out runs/exp4_temporal/flicker_curve.png || true
