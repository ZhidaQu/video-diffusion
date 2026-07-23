#!/usr/bin/env bash
# EXP2 - Hyperparameter sensitivity (paper: sensitivity study). One benchmark,
# three sweeps: temporal blend alpha, diffusion steps, detector box threshold.
source "$(dirname "$0")/_env.sh"
B=data/avenue_night

echo "== 2a: temporal blend alpha (GT mask + hf_diffusion temporal) =="
queue \
 "--benchmark $B --detector gt_mask --inpainter hf_diffusion --temporal --blend 0.2  --tag e2_blend_020" \
 "--benchmark $B --detector gt_mask --inpainter hf_diffusion --temporal --blend 0.35 --tag e2_blend_035" \
 "--benchmark $B --detector gt_mask --inpainter hf_diffusion --temporal --blend 0.5  --tag e2_blend_050" \
 "--benchmark $B --detector gt_mask --inpainter hf_diffusion --temporal --blend 0.65 --tag e2_blend_065" \
 "--benchmark $B --detector gt_mask --inpainter hf_diffusion --temporal --blend 0.8  --tag e2_blend_080"
python scripts/table.py --benchmark "$B" \
 --tags e2_blend_020,e2_blend_035,e2_blend_050,e2_blend_065,e2_blend_080 \
 --out runs/exp2_hyperparam/blend.md

echo "== 2b: diffusion sampling steps =="
queue \
 "--benchmark $B --detector gt_mask --inpainter hf_diffusion --temporal --steps 15 --tag e2_steps_15" \
 "--benchmark $B --detector gt_mask --inpainter hf_diffusion --temporal --steps 25 --tag e2_steps_25" \
 "--benchmark $B --detector gt_mask --inpainter hf_diffusion --temporal --steps 35 --tag e2_steps_35" \
 "--benchmark $B --detector gt_mask --inpainter hf_diffusion --temporal --steps 50 --tag e2_steps_50"
python scripts/table.py --benchmark "$B" \
 --tags e2_steps_15,e2_steps_25,e2_steps_35,e2_steps_50 \
 --out runs/exp2_hyperparam/steps.md

echo "== 2c: detector box threshold (grounded_sam2 + bg_diffusion) =="
queue \
 "--benchmark $B --detector grounded_sam2 --box-threshold 0.20 --text-threshold 0.20 --dilate 5 --inpainter bg_diffusion --temporal --tag e2_bt_020" \
 "--benchmark $B --detector grounded_sam2 --box-threshold 0.25 --text-threshold 0.20 --dilate 5 --inpainter bg_diffusion --temporal --tag e2_bt_025" \
 "--benchmark $B --detector grounded_sam2 --box-threshold 0.30 --text-threshold 0.20 --dilate 5 --inpainter bg_diffusion --temporal --tag e2_bt_030" \
 "--benchmark $B --detector grounded_sam2 --box-threshold 0.35 --text-threshold 0.20 --dilate 5 --inpainter bg_diffusion --temporal --tag e2_bt_035"
python scripts/table.py --benchmark "$B" \
 --tags e2_bt_020,e2_bt_025,e2_bt_030,e2_bt_035 \
 --out runs/exp2_hyperparam/box_threshold.md
