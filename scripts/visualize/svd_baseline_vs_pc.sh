#!/bin/bash
# Dump singular values for BOTH 1B runs that the plots compare:
#   - baseline (no PC)        -> visualization_output/svd/llama2_1B_AdamW_AdamW_baseline_1B/
#   - PC on ffn + o blocks    -> visualization_output/svd/llama2_1B_AdamW_AdamW_pc_1B/
#
# Run this first, then feed the two output dirs to plot_gmcn.sh /
# plot_sv_hist.sh. Override STEPS / NGPU via env if needed
# (STEPS=-1 dumps every checkpoint step; STEPS=61100 dumps a single step).
set -ex

# Activate your environment first (example with conda):
#   conda activate pc
# This script cd's to the repo root so the relative config/output paths resolve.
cd "$(dirname "$0")/../.." || exit 1

export USE_LIBUV=1
export CC=gcc

CONFIG_ROOT=./train_configs
# Both configs: baseline (precondition_* = false) and PC on ffn+o.
CONFIG_FILES=(
    "${CONFIG_ROOT}/AdamW_baseline_1B.toml"
    "${CONFIG_ROOT}/AdamW_pc_1B.toml"
)
STEPS=${STEPS:--1}
NGPU=${NGPU:-1}

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
        --local-ranks-filter 0 --role rank --tee 3 \
        train.py --job.config_file "${CONFIG_FILE}" \
            --visualize.enable \
            --visualize.step "${STEPS}"
done
