#!/bin/zsh
# Compare the existing baseline/03_cnn_vit_cnn against baseline/03b_cnn_vit_cnn_enhanced
# (restrict_range=clamp, refine_kernel_size=5, use_motion_channels=true, tuned lr/wd --
# see EXPERIMENT_NOTES.md and icenet_mp/config/baseline/03b_cnn_vit_cnn_enhanced.yaml
# for what each change is and why).
#
# Nothing in this script runs automatically -- call it with exactly one of the stage
# names below. Each stage is a real training run; none are dry-runs.
#
#   ./scripts/cnn_vit_cnn_comparison.sh synthetic-old
#   ./scripts/cnn_vit_cnn_comparison.sh synthetic-new
#   ./scripts/cnn_vit_cnn_comparison.sh real-calibrate
#   ./scripts/cnn_vit_cnn_comparison.sh real-old
#   ./scripts/cnn_vit_cnn_comparison.sh real-new
#
# synthetic-{old,new}: `imp synthetic check` at --grid-size 432 -- i.e. the real
#   full_north grid resolution, but on the fast synthetic moving-circle data instead of
#   real data. Smoke test that the enhanced config (motion channels doubling every
#   encoder's input channels, clamp, refine_kernel_size=5) builds and trains sensibly
#   at real scale, before spending real-data compute on it.
#
# real-calibrate: 2 real epochs on the enhanced config against `full_north_from_1999`
#   (data/full_north_from_1999.yaml: same as full_north but training starts 1999, not
#   1979 -- float-argo has no data before 1999 anyway). Purely to get real wall-clock
#   per-epoch timing and catch real-data-loading issues before committing to the full
#   run; not a result to read anything into.
#
# real-{old,new}: the actual comparison, against full_north_from_1999 at native
#   432x432 resolution. max_epochs overridden to 200 (train/trainer/default.yaml's
#   own default is 50) together with the existing early_stopping callback
#   (patience=20 on validation_loss) and best_checkpoint -- i.e. it runs up to 200
#   epochs but will stop up to 20 epochs after validation_loss last improved, and the
#   checkpoint kept is the best one regardless of when it stops. EMA weight averaging
#   is disabled for all real-* stages (decay_rate=0.999 needs ~1000+ updates to
#   converge away from a random-init average; with early_stopping's patience meaning
#   we don't know in advance how many epochs either run gets, EMA convergence would
#   be an uncontrolled, differential confound between old and new -- same reason
#   every cnn_vit_cnn run in EXPERIMENT_NOTES disables it).
#
#   real-old/real-new publish to W&B (entity 'turing-seaice', the shared team
#   workspace) in addition to the local report, with explicit run names so the two
#   are easy to tell apart there. real-calibrate stays local-only -- it's a throwaway
#   2-epoch timing check, not a result worth publishing.

set -eu
cd "$(dirname "$0")/.."

BASE_PATH=/Volumes/Storage/ClimateData
DATA=full_north_from_1999

case "${1:-}" in
  synthetic-old)
    mkdir -p outputs/synthetic_check_atscale432_cnn_vit_cnn_old
    uv run imp synthetic check \
      data=synthetic predict=synthetic-2d train=synthetic evaluate=synthetic \
      'loggers=[local_files]' \
      model.decoder.mask_type=none \
      random.seed=1234 \
      '~train.callbacks.ema_weight_averaging' \
      +train.callbacks.best_checkpoint.save_last=true \
      --config-name baseline/03_cnn_vit_cnn \
      --output-dir outputs/synthetic_check_atscale432_cnn_vit_cnn_old \
      --grid-size 432 --n-trajectories 16 --max-epochs 20
    ;;
  synthetic-new)
    mkdir -p outputs/synthetic_check_atscale432_cnn_vit_cnn_enhanced
    uv run imp synthetic check \
      data=synthetic predict=synthetic-2d train=synthetic evaluate=synthetic \
      'loggers=[local_files]' \
      model.decoder.mask_type=none \
      random.seed=1234 \
      '~train.callbacks.ema_weight_averaging' \
      +train.callbacks.best_checkpoint.save_last=true \
      --config-name baseline/03b_cnn_vit_cnn_enhanced \
      --output-dir outputs/synthetic_check_atscale432_cnn_vit_cnn_enhanced \
      --grid-size 432 --n-trajectories 16 --max-epochs 20
    ;;
  real-calibrate)
    # EMA and early_stopping both disabled: EMA (decay_rate=0.999, updates every
    # 100 steps AND every epoch) needs ~1000+ updates to substantially converge away
    # from its random-init starting average; 2 epochs gives ~77 updates -- reading
    # a validation loss computed against that near-unconverged EMA average would be
    # meaningless. early_stopping's patience=20 can't fire in 2 epochs anyway.
    uv run imp train \
      --config-name baseline/03b_cnn_vit_cnn_enhanced \
      "data=$DATA" predict=sic-ssmis-14d \
      'loggers=[local_files]' \
      "++base_path=$BASE_PATH" \
      '~train.callbacks.early_stopping' \
      '~train.callbacks.ema_weight_averaging' \
      train.trainer.max_epochs=2
    ;;
  real-old)
    # Each run gets its own timestamped/hashed directory under
    # $BASE_PATH/training/local/ automatically (same pattern as the synthetic check
    # outputs) -- old and new runs sharing $BASE_PATH will not collide.
    #
    # EMA disabled: with early_stopping patience=20, we don't know in advance how
    # many epochs this will actually run for, and EMA's ~1000-update time constant
    # means "old" and "new" could end up differentially converged (and thus not
    # fairly comparable) depending purely on when each happens to stop -- the same
    # reason every cnn_vit_cnn run in EXPERIMENT_NOTES disables it.
    uv run imp train \
      --config-name baseline/03_cnn_vit_cnn \
      "data=$DATA" predict=sic-ssmis-14d \
      'loggers=[wandb,local_files]' \
      loggers.wandb.name=cnn_vit_cnn_old_full_north_1999 \
      "++base_path=$BASE_PATH" \
      '~train.callbacks.ema_weight_averaging' \
      train.trainer.max_epochs=200
    ;;
  real-new)
    uv run imp train \
      --config-name baseline/03b_cnn_vit_cnn_enhanced \
      "data=$DATA" predict=sic-ssmis-14d \
      'loggers=[wandb,local_files]' \
      loggers.wandb.name=cnn_vit_cnn_enhanced_full_north_1999 \
      "++base_path=$BASE_PATH" \
      '~train.callbacks.ema_weight_averaging' \
      train.trainer.max_epochs=200
    ;;
  *)
    echo "Usage: $0 {synthetic-old|synthetic-new|real-calibrate|real-old|real-new}" >&2
    exit 1
    ;;
esac
