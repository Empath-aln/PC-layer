#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Launch training for the AdamW_pc_1B configuration.
#
# Activate your environment first (example with conda):
#   conda activate pc
# then run this script from anywhere; it cd's to the repo root so that the
# relative paths in the .toml resolve correctly.
cd "$(dirname "$0")/../.." || exit 1

set -ex

# libUV is a scalable backend for TCPStore used in processGroup rendezvous.
export USE_LIBUV=1
export CC=gcc

# Experiment logging via SwanLab. Export your own key before launching, e.g.
#   export SWANLAB_API_KEY=<your-key>
# To use Weights & Biases instead, see the README ("Experiment logging").
export SWANLAB_DIR=${SWANLAB_DIR:-./swanlab}

# Local overrides for convenience, e.g.
#   LOG_RANK=0,1 NGPU=4 bash scripts/train/AdamW_pc_1B.sh
NGPU=${NGPU:-"8"}
NNODES=${NNODES:-"1"}

# By default log just rank 0 output.
LOG_RANK=${LOG_RANK:-0}

CONFIG_FILE=${CONFIG_FILE:-"./train_configs/AdamW_pc_1B.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} $overrides
