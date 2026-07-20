#!/usr/bin/env bash
# Train cnn_vit_cnn on SIC (SSMIS) input only -- no ERA5/Argo -- predicting 14 days.
# Dropping the extra input datasets, and selecting only the ice_conc variable from
# SSMIS (the other 5 channels are a raw duplicate, NaN-filled uncertainties and a
# status flag), shrinks the latent from 72 to 2 channels: measured ~3.2 it/s vs the
# original 0.64 it/s, i.e. ~20 min/epoch instead of ~91 (see speed_issues.md).
# Early stopping (patience 20 on validation_loss) decides when to finish.
set -euo pipefail

RUN_NAME="${RUN_NAME:-cnn_vit_cnn_sic_only_full_north_1999}"
BASE_PATH="${BASE_PATH:-/Volumes/Storage/ClimateData}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
SPLIT="${SPLIT:-full_dataset_from_1999}"

# Any extra arguments are passed through as additional hydra overrides.
uv run imp train \
    --config-name baseline/03_cnn_vit_cnn \
    data=full_north_from_1999 \
    "data/split=${SPLIT}" \
    'data/datasets=[full_sicnorth_ssmis_25p0km_1979_2024_24h_v2]' \
    '+data.datasets.full-sicnorth-ssmis-25p0km-1979-2024-24h-v2.variables=[ice_conc]' \
    predict=sic-ssmis-14d \
    'loggers=[wandb,local_files]' \
    loggers.wandb.name="${RUN_NAME}" \
    ++base_path="${BASE_PATH}" \
    '~train.callbacks.ema_weight_averaging' \
    train.trainer.max_epochs="${MAX_EPOCHS}" \
    "$@"
