#!/usr/bin/env bash
# A/B comparison of the two speed-oriented training splits, run sequentially
# (one GPU), 50 max epochs each, SIC-only ice_conc input (see train_sic_only.sh):
#   1. recent_8yr        -- 8 recent training years,      ~1460 samples/epoch
#   2. stride2_from_1999 -- all 22 years, window stride 2, ~1880 samples/epoch
# Both validate on 2021 only, so their validation curves are directly comparable.
set -euo pipefail
cd "$(dirname "$0")"

MAX_EPOCHS="${MAX_EPOCHS:-50}"

echo "=== Run 1/2: recent_8yr ==="
SPLIT=recent_8yr RUN_NAME=cnn_vit_cnn_sic_recent8 MAX_EPOCHS="${MAX_EPOCHS}" \
    ./train_sic_only.sh

echo "=== Run 2/2: stride2_from_1999 ==="
SPLIT=stride2_from_1999 RUN_NAME=cnn_vit_cnn_sic_stride2 MAX_EPOCHS="${MAX_EPOCHS}" \
    ./train_sic_only.sh
