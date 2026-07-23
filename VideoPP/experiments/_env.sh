#!/usr/bin/env bash
# Common environment + a small dual-GPU dispatch helper. Source at the top of
# every experiment script:  source "$(dirname "$0")/_env.sh"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Model weights live inside the repo (on /data), never the home dir.
export HF_HOME="$REPO/weights/hf"
export TORCH_HOME="$REPO/weights/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# GPUs to use. Edit for your machine.
GPUS=(0 1)

# Activate the conda env holding torch/transformers/diffusers. Edit to yours.
# source ~/.conda/etc/profile.d/conda.sh && conda activate avr

# queue "<run_one args>" "<run_one args>" ... : dispatch across $GPUS, N at a time.
queue() {
  local i=0 pids=()
  for job in "$@"; do
    local gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    echo ">> gpu$gpu  run_one.py $job"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/run_one.py $job &
    pids+=($!); i=$((i + 1))
    if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait "${pids[@]}" || true; pids=(); fi
  done
  [ ${#pids[@]} -gt 0 ] && { wait "${pids[@]}" || true; }
}
