# Run an evaluation job

This guide walks through running an evaluation on a trained checkpoint: launching the job, enabling visualisations, saving predictions, and finding the outputs.

## Prerequisites

Make sure IceNet-MP is [installed](../user-guide/installation.md) and that you have a trained checkpoint — either from a [training run](train.md) or downloaded from shared storage.

## 1. Get a checkpoint

### From a training run

Checkpoints are saved to `${BASE_DIR}/training/wandb/run-<date>-<id>/checkpoints/<name>.ckpt` after training. Pick the checkpoint you want to evaluate.

### From shared storage (HPC)

Pre-trained checkpoints are available on Baskerville, DAWN, and Isambard-AI.
Ask a team member for the path.

## 2. Create a local config

If you do not already have a local config from a training run, create one at `icenet_mp/config/<your-name>.local.yaml`.
See [Train a model — Create a local config](train.md#2-create-a-local-config) for details.

## 3. Run evaluate

See the [`evaluate` command reference](../user-guide/commands.md#evaluate) for full option details, then run:

```bash
uv run imp evaluate --config-name <your-name>.local --checkpoint PATH_TO_CHECKPOINT
```

### Enabling visualisations

By default, all visualisations are enabled (see `icenet_mp/config/evaluate/callbacks/plotting.yaml`). To disable forecast plots, set `make_static_plots` and `make_video_plots` to `false` in your local config:

```yaml
evaluate:
  callbacks:
    plotting:
      make_static_plots: false
      make_video_plots: false
```

Plots of the raw input data are also enabled by default. To disable them, set:

```yaml
evaluate:
  callbacks:
    plotting:
      make_input_plots: false
```

### Saving predictions as NetCDF

Pass `--save-predictions` to write the model output from the configured test period to a NetCDF file:

```bash
uv run imp evaluate \
  --config-name <your-name>.local \
  --checkpoint PATH_TO_CHECKPOINT \
  --save-predictions predictions.nc
```

The file is written incrementally during evaluation, so the full test period does not need to be held in memory. It contains `forecast_reference_time`, `lead_time`, `valid_time`, latitude/longitude coordinates, and one data variable per prediction target. Sea-ice concentration is exported as `ice_conc` in its original source scale with CF `sea_ice_area_fraction` metadata.

Prediction export uses the existing `data.split.test` date ranges. To export a smaller date range, change the test split in the config rather than running a separate prediction pass.

NetCDF export currently supports single-process evaluation. If the evaluation config uses multiple devices, set `evaluate.trainer.devices=1` for the export run.

## 4. Check results in W&B

Once evaluation completes, the run appears in the W&B project `evaluate` under the `turing-seaice` entity at [wandb.ai](https://wandb.ai).

| Key | Contents |
|-----|----------|
| `output_static` | Static images of forecast output. |
| `output_video` | Animated forecast output. |
| `input_static` | Static images of the raw input data (if `make_input_plots: true`). |
| `input_video` | Animated raw input data (if `make_input_plots: true`). |
| `Custom Charts` | Per-forecast-day metrics, allowing skill to be assessed at longer lead times. |
