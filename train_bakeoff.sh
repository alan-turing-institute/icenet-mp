#!/usr/bin/env bash
# Blur bake-off (see next_steps.md), run sequentially (one GPU), 50 max epochs each.
# All runs: identical data to the old-vs-new comparison -- SIC (SSMIS) trimmed to
# ice_conc only, recent_8yr split (train 2014-18/2020/2022/2024, validate 2021,
# test 2019 untouched), predict 14 days, EMA callback removed, wandb + local_files.
#
#   1. cnn_vit_cnn_residual  -- baseline/03c: the enhanced CNN-ViT-CNN plus skip
#      connection and residual (persistence-delta) output.
#   2. cnn_unet_cnn_enhanced -- baseline/02b: same enhancement set on the
#      cnn-unet-cnn arrangement, GroupNorm throughout (BatchNorm collapses the UNet
#      processor at batch ~2). Only the processor differs vs run 1.
#   3. unet_old_style        -- baseline/01_unet unchanged: IceNet-style UNet at full
#      resolution, sigmoid range restriction, no enhancements. Known risk: BatchNorm
#      train/eval divergence at batch 2 can collapse it; if that happens, rerun with
#      model.processor.norm_type=groupnorm and note the deviation.
#
# The persistence numbers to beat on the 2021 validation year are in next_steps.md
# (mean RMSE 0.0422; models must win on days 8-14 skill, not just mean loss).
set -euo pipefail
cd "$(dirname "$0")"

MAX_EPOCHS="${MAX_EPOCHS:-50}"
BASE_PATH="${BASE_PATH:-/Volumes/Storage/ClimateData}"

DATA_ARGS=(
    data=full_north_from_1999
    data/split=recent_8yr
    'data/datasets=[full_sicnorth_ssmis_25p0km_1979_2024_24h_v2]'
    '+data.datasets.full-sicnorth-ssmis-25p0km-1979-2024-24h-v2.variables=[ice_conc]'
    predict=sic-ssmis-14d
    'loggers=[wandb,local_files]'
    ++base_path="${BASE_PATH}"
    '~train.callbacks.ema_weight_averaging'
    train.trainer.max_epochs="${MAX_EPOCHS}"
)

echo "=== Run 1/3: cnn_vit_cnn_residual (baseline/03c) ==="
uv run imp train \
    --config-name baseline/03c_cnn_vit_cnn_residual \
    "${DATA_ARGS[@]}" \
    loggers.wandb.name=cnn_vit_cnn_residual_sic_recent8

echo "=== Run 2/3: cnn_unet_cnn_enhanced (baseline/02b) ==="
uv run imp train \
    --config-name baseline/02b_cnn_unet_cnn_enhanced \
    "${DATA_ARGS[@]}" \
    loggers.wandb.name=cnn_unet_cnn_enhanced_sic_recent8

echo "=== Run 3/3: unet_old_style (baseline/01_unet) ==="
uv run imp train \
    --config-name baseline/01_unet \
    "${DATA_ARGS[@]}" \
    loggers.wandb.name=unet_old_style_sic_recent8
