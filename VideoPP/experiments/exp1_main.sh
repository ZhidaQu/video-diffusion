#!/usr/bin/env bash
# EXP1 - Main experiment. Two tables, following the video-inpainting convention.
#
# TABLE 1 (fixed-mask restoration fidelity): every method fills the SAME GT
#   mask, so the comparison isolates restoration quality. mask_IoU is NOT reported
#   here (all methods share the mask). Independent published baselines must be
#   installed and registered (see bottom); classical OpenCV/flow baselines run now.
# TABLE 2 (end-to-end systems): full detect->remove->inpaint pipelines with their
#   own masks. Naive frame-wise Grounded-SAM+SD vs our pipeline.
#
# Published numbers are NOT reusable: no prior work reports restoration fidelity
# on Ped2/Avenue/ShanghaiTech under an anomaly-injection-with-clean-GT protocol,
# and inpainting papers report on DAVIS/YouTube-VOS. Baselines are RUN here.
source "$(dirname "$0")/_env.sh"

BENCHES=(data/ped2_night data/avenue_night data/sht_night)

for B in "${BENCHES[@]}"; do
  echo "==== EXP1 Table 1 (fixed GT mask) on $B ===="
  queue \
    "--benchmark $B --detector gt_mask --inpainter spatial_cv2 --tag e1t1_classical_spatial" \
    "--benchmark $B --detector gt_mask --inpainter flow_fill    --tag e1t1_classical_flow" \
    "--benchmark $B --detector gt_mask --inpainter propainter   --tag e1t1_propainter" \
    "--benchmark $B --detector gt_mask --inpainter hf_diffusion --temporal --tag e1t1_ours_diffusion" \
    "--benchmark $B --detector gt_mask --inpainter bg_diffusion --temporal --tag e1t1_ours_bg"
  python scripts/table.py --benchmark "$B" \
    --tags e1t1_classical_spatial,e1t1_classical_flow,e1t1_propainter,e1t1_ours_diffusion,e1t1_ours_bg \
    --out "runs/exp1_main/table1_$(basename "$B").md"

  echo "==== EXP1 Table 2 (end-to-end systems) on $B ===="
  queue \
    "--benchmark $B --detector grounded_sam  --box-threshold 0.25 --text-threshold 0.20 --dilate 5 --inpainter hf_diffusion --tag e1t2_grounded_sd_naive" \
    "--benchmark $B --detector grounded_sam2 --box-threshold 0.25 --text-threshold 0.20 --dilate 5 --inpainter bg_diffusion --temporal --tag e1t2_ours_fullpipeline"
  python scripts/table.py --benchmark "$B" \
    --tags e1t2_grounded_sd_naive,e1t2_ours_fullpipeline \
    --out "runs/exp1_main/table2_$(basename "$B").md"
done

# ---- More independent baselines: see BASELINES.md for status + setup.
# ProPainter is already active above. E2FGVI is written (avr/inpaint/e2fgvi.py) but
# needs a separate mmcv env; after setting it up, add this line to Table 1:
#   "--benchmark $B --detector gt_mask --inpainter e2fgvi --tag e1t1_e2fgvi"
# DiffuEraser / OmnimatteZero: integration recipe in BASELINES.md.
