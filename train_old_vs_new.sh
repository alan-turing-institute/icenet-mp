#!/usr/bin/env bash
# Old-vs-new model comparison, run sequentially (one GPU), 50 max epochs each.
# Both runs: SIC (SSMIS) input only, recent_8yr split (8 recent years, validate on
# 2021), predict 14 days, EMA callback removed, logged to wandb.
#
#   1. OLD -- origin/main's baseline/03_cnn_vit_cnn, run from the main worktree at
#      ../icenet-mp-main: per-patch linear ViT decode, no motion/day-order channels,
#      sigmoid restrict_range, default lr. Main cannot select input variables, so it
#      sees all 6 SSMIS channels.
#   2. NEW -- this branch's baseline/03b_cnn_vit_cnn_enhanced: conv refinement head
#      (k=5), motion channels, day-order channels, clamp restrict_range, tuned
#      lr/weight_decay, input trimmed to the ice_conc variable only.
set -euo pipefail
cd "$(dirname "$0")"

MAX_EPOCHS="${MAX_EPOCHS:-50}"
BASE_PATH="${BASE_PATH:-/Volumes/Storage/ClimateData}"
MAIN_WORKTREE="${MAIN_WORKTREE:-/Users/aoife/git/icenet-mp-main}"

echo "=== Run 1/2: OLD (main baseline/03_cnn_vit_cnn) ==="
(
    cd "${MAIN_WORKTREE}"
    uv run imp train \
        --config-name baseline/03_cnn_vit_cnn \
        data=full_north \
        data/split=recent_8yr \
        'data/datasets=[full_sicnorth_ssmis_25p0km_1979_2024_24h_v2]' \
        predict=sic-ssmis-14d \
        'loggers=[wandb]' \
        loggers.wandb.name=cnn_vit_cnn_old_main_sic_recent8 \
        ++base_path="${BASE_PATH}" \
        '~train.callbacks.ema_weight_averaging' \
        train.trainer.max_epochs="${MAX_EPOCHS}"
)

echo "=== Run 2/2: NEW (branch baseline/03b_cnn_vit_cnn_enhanced) ==="
uv run imp train \
    --config-name baseline/03b_cnn_vit_cnn_enhanced \
    data=full_north_from_1999 \
    data/split=recent_8yr \
    'data/datasets=[full_sicnorth_ssmis_25p0km_1979_2024_24h_v2]' \
    '+data.datasets.full-sicnorth-ssmis-25p0km-1979-2024-24h-v2.variables=[ice_conc]' \
    predict=sic-ssmis-14d \
    'loggers=[wandb,local_files]' \
    loggers.wandb.name=cnn_vit_cnn_enhanced_sic_recent8 \
    ++base_path="${BASE_PATH}" \
    '~train.callbacks.ema_weight_averaging' \
    train.trainer.max_epochs="${MAX_EPOCHS}"
