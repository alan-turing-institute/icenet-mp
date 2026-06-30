# Run an evaluation job

This guide walks through running an evaluation on a trained checkpoint: launching the job, enabling visualisations, and finding the outputs.

## Prerequisites

Make sure IceNet-MP is [installed](../user-guide/installation.md) and that you have a trained checkpoint — either from a [training run](run-training.md) or downloaded from shared storage.

## 1. Get a checkpoint

### From a training run

Checkpoints are saved to `${BASE_DIR}/training/wandb/run-<date>-<id>/checkpoints/<name>.ckpt` after training. Pick the checkpoint you want to evaluate.

### From shared storage (HPC)

Pre-trained checkpoints are available on Baskerville, DAWN, and Isambard-AI.
Ask a team member for the path.

## 2. Create a local config

If you do not already have a local config from a training run, create one at `icenet_mp/config/<your-name>.local.yaml`.
See [Run a training job — Create a local config](run-training.md#2-create-a-local-config) for details.

## 3. Run evaluate

See the [`evaluate` command reference](../user-guide/commands.md#evaluate) for full option details, then run:

```bash
uv run imp evaluate --config-name <your-name>.local --checkpoint PATH_TO_CHECKPOINT
```

### Enabling visualisations

By default, visualisations are enabled (see `evaluate/callbacks/plotting.yaml`). To disable them, set `make_static_plots` and `make_video_plots` to `false` in your local config:

```yaml
evaluate:
  callbacks:
    plotting:
      make_static_plots: false
      make_video_plots: false
```

To enable plots of the raw input data alongside the forecasts, add:

```yaml
evaluate:
  callbacks:
    plotting:
      make_input_plots: true
```

Output directories, styling, and animation parameters are set under `evaluate.callbacks.plotting.plot_spec` and can be overridden at the command line.

## 4. Check results in W&B

Once evaluation completes, the run appears in the `turing-seaice` W&B project at [wandb.ai](https://wandb.ai).

| Key | Contents |
|-----|----------|
| `output_static` | Static images of forecast output. |
| `output_video` | Animated forecast output. |
| `input_static` | Static images of the raw input data (if `make_input_plots: true`). |
| `input_video` | Animated raw input data (if `make_input_plots: true`). |
| `Custom Charts` | Per-forecast-day metrics, allowing skill to be assessed at longer lead times |
