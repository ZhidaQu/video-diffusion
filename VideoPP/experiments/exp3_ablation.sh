#!/usr/bin/env bash
# EXP3 - Ablation (paper: ablation study). Remove one component at a time from our
# full method = grounded_sam2 + bg_diffusion + temporal.
source "$(dirname "$0")/_env.sh"
B=data/avenue_night
D="--box-threshold 0.25 --text-threshold 0.20 --dilate 5"

queue \
 "--benchmark $B --detector grounded_sam2 $D --inpainter bg_diffusion --temporal --tag e3_full" \
 "--benchmark $B --detector grounded_sam2 $D --inpainter bg_diffusion            --tag e3_wo_temporal" \
 "--benchmark $B --detector grounded_sam2 $D --inpainter hf_diffusion --temporal --tag e3_wo_bg" \
 "--benchmark $B --detector grounded_sam2 $D --inpainter hf_diffusion            --tag e3_wo_bg_wo_temporal" \
 "--benchmark $B --detector grounded_sam   $D --inpainter bg_diffusion --temporal --tag e3_wo_sam2"
python scripts/table.py --benchmark "$B" \
 --tags e3_full,e3_wo_temporal,e3_wo_bg,e3_wo_bg_wo_temporal,e3_wo_sam2 \
 --out runs/exp3_ablation/ablation.md
