# Run an evaluation job

This guide walks through running an evaluation on a trained checkpoint: launching the job, enabling visualisations, and finding the outputs.

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

## 4. Check results in W&B

Once evaluation completes, the run appears in the W&B project `evaluate` under the `turing-seaice` entity at [wandb.ai](https://wandb.ai).

| Key | Contents |
|-----|----------|
| `output_static` | Static images of forecast output. |
| `output_video` | Animated forecast output. |
| `input_static` | Static images of the raw input data (if `make_input_plots: true`). |
| `input_video` | Animated raw input data (if `make_input_plots: true`). |
| `Custom Charts` | Per-forecast-day metrics, allowing skill to be assessed at longer lead times. |

### How to interpret forecast days and example videos

The per-forecast-day charts and the example videos summarise different dimensions of the same evaluation run.

For each valid start date in the configured evaluation period, the model produces `n_forecast_steps` lead times. A metric shown for forecast day 1, 2, 3, and so on is aggregated across all evaluation start dates at that lead time. For example, the value at forecast day 3 answers: "how accurate was the third forecast step, averaged over the evaluation period?"

The videos are not one frame per evaluation start date. They use a small set of representative start dates sampled across the evaluation period, and each video contains the forecast sequence for that selected start date. As a result, it is normal for a run to show many forecast-day metric points while only producing a few example videos.

In short:

- `Custom Charts` show performance by lead time, aggregated across the evaluation period.
- `output_video` shows a small number of representative forecast sequences from within that period.

This distinction is useful when checking whether skill degrades with lead time without generating a video for every possible forecast start date.
