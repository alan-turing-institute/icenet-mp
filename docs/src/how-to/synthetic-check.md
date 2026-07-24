# Run the synthetic pipeline check

The synthetic pipeline check trains a real, unmodified model configuration on generated moving-shape data (bouncing circles) instead of real sea-ice data, then evaluates it on held-out trajectories.
It exercises the full pipeline — data loading, training, checkpointing, evaluation, plotting — in minutes rather than hours, so it is the fastest way to sanity-check a model or architecture change before committing to a real-data run.

Whole trajectories are held out for validation and test, so passing requires the model to learn the general translate-and-bounce rule, not memorise a single sequence.
The check passes if validation loss improves by at least `--min-relative-improvement` (default 30%) from the first epoch to the best epoch.

## Prerequisites

Make sure IceNet-MP is [installed](../user-guide/installation.md).
No real data, base path, or W&B account is needed: the check generates its own dataset and logs to local files.

## Running the check

Two baseline configurations are provided, one per architecture.

### UNet

```bash
# Small (32x32) -- quickest smoke test, ~a few minutes
uv run imp synthetic-check --config-name baseline/synthetic_unet

# Midsize (144x144)
uv run imp synthetic-check --config-name baseline/synthetic_unet --grid-size 144

# Full size (432x432) -- matches the real-data resolution
uv run imp synthetic-check --config-name baseline/synthetic_unet --grid-size 432
```

### CNN-ViT-CNN

```bash
# Small (48x48) -- the smallest grid this model supports (see notes below)
uv run imp synthetic-check --config-name baseline/synthetic_cnn_vit --grid-size 48

# Midsize (144x144)
uv run imp synthetic-check --config-name baseline/synthetic_cnn_vit --grid-size 144

# Full size (432x432)
uv run imp synthetic-check --config-name baseline/synthetic_cnn_vit --grid-size 432
```

Each run writes the generated dataset, checkpoints, loss-curve and prediction plots, and a pass/fail report to `--output-dir` (default `outputs/synthetic_check`).

### Choosing the dynamics

The `--dynamics` option selects what the synthetic shapes do, so you can exercise different failure modes:

```bash
# Advection: a rigid circle translates and bounces off the edges (the default).
uv run imp synthetic-check --config-name baseline/synthetic_unet --dynamics moving

# Growth/melt: a stationary blob grows and shrinks in place via a morphological
# open/close cycle, mimicking sea ice advancing and retreating seasonally.
uv run imp synthetic-check --config-name baseline/synthetic_unet --dynamics grow-shrink
```

`moving` tests whether the model learns to translate a fixed shape; `grow-shrink` tests whether it learns concentration change in place (the shape never moves, but its extent pulses). Both work with either baseline and at any valid grid size.

## Notes and constraints

- **Grid sizes must respect each architecture's minimum.**
  Every grid size must be a multiple of 16 and larger than 16: the UNet processor pools the grid four times, and the check applies this constraint to all models.
  The ViT processor additionally patchifies the grid with `patch_size: 24`, so for `synthetic_cnn_vit` the grid must also be a multiple of 24 — in practice, use a multiple of 48 (the least common multiple). The default `--grid-size 32` therefore works for the UNet but **not** for the CNN-ViT, whose smallest working grid is 48.
  *(The ViT's patch size can be changed via config or CLI overrides if you need a different divisor; adjust both `patch_size` and the grid size accordingly.)*
- **`encoders.latent_space` is set automatically** to match `--grid-size`, so the model works at any (valid) resolution without editing configs. This reshapes the working resolution only, not the model's capacity.
- **"No land mask available" warnings are expected.** The synthetic world has no land, so the baseline configs set `model.decoder.mask_type: none`, and plotting falls back accordingly at non-432 shapes.
- **Runtime scales roughly with the square of the grid size.** Use small grids to iterate and 432 only to confirm behaviour at the real resolution; `--max-epochs` caps the run length (with few epochs, you may also need to lower `--min-relative-improvement` since the pass gate is calibrated for a full run).
- **Give the UNet enough epochs before judging it.** Its BatchNorm layers use different statistics in train and eval mode, so validation loss often stays flat for the first ~5 epochs while training loss falls, then catches up as the running statistics settle. A 2–5 epoch run can therefore fail the improvement gate spuriously; the default 30 epochs (with early stopping) is reliable.
- **Passing is necessary, not sufficient.** The synthetic shapes move deterministically, so a model that cannot learn them has no chance on real ice — but success here does not guarantee skill on real data.
- **CI runs a small version automatically for both architectures.** The `Synthetic pipeline check` workflow runs a matrix on every pull request: a narrow UNet (`model.processor.start_out_channels=32`, 12 epochs, 32×32 grid) and the CNN-ViT-CNN (6 epochs, 48×48 grid). Each arm runs the check end-to-end and additionally asserts the best validation RMSE stays under a lenient per-model threshold — see `.github/workflows/synthetic_pipeline_check.yaml`.
