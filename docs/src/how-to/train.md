# Train a model

This guide walks through an end-to-end training run: getting data, creating a local config, launching training, and inspecting results in Weights & Biases.
Single-stage training trains the full model end-to-end in one pass.
It is the default and works for all model architectures.

## Prerequisites

Make sure IceNet-MP is [installed](../user-guide/installation.md) before continuing.

You will need a [Weights & Biases account](https://docs.wandb.ai/models/quickstart).
Generate an API key, then authenticate before running any training command:

```bash
export WANDB_API_KEY=<your_api_key>
wandb login
```

## 1. Get data

### Using pre-downloaded data (HPC)

If you are working on Baskerville, DAWN, or Isambard-AI, the datasets are already available on shared storage. Use the appropriate base config and skip ahead to [step 2](#2-create-a-local-config):

```yaml
defaults:
  - base_isambardai   # or base_baskerville or base_dawn
  - _self_
```

### Downloading data locally

See the [`datasets create` command reference](../user-guide/commands.md#datasets-create) for prerequisites and usage.

## 2. Create a local config

Create a file at `icenet_mp/config/<your-name>.local.yaml`. The base config you inherit from depends on where you are running.

### On Isambard-AI

Inherit from `base_isambardai`, which already points to the shared data storage:

```yaml
defaults:
  - base_isambardai
  - _self_
```

Use `base_baskerville` or `base_dawn` instead if you are on one of those systems.

### On a local machine

Inherit from `base` and set `base_path` to the directory where you downloaded your data:

```yaml
defaults:
  - base
  - _self_

base_path: /path/to/your/data
```

See [Configuration](../user-guide/configuration.md) for how to switch datasets or override model parameters.

### Reusing a config from a previous W&B run

To reproduce or extend a prior run, download its saved config from the W&B run page under **Files > `model_config.yaml`** and place it at `icenet_mp/config/<your-name>.local.yaml`.
Make sure that the `defaults` block points at the appropriate base config for your machine:

```yaml
defaults:
  - base_isambardai   # or base_baskerville, base_dawn, base.local etc.
  - _self_

# ... rest of the downloaded config ...
```

If you are running locally, ensure that `base_path` is set as well.

## 3. Run training

```bash
uv run imp train --config-name <your-name>.local
```

## Configuring training

Training is controlled by the `train` section of your config.
The most commonly adjusted settings are:

```yaml
train:
  optimizer:
    lr: 1e-3
    weight_decay: 1e-4
  scheduler:
    T_max: 20
    eta_min: 1e-5
  trainer:
    max_epochs: 20
    accelerator: auto
```

By default, the models use an active grid cell mask in the decoder, which sets all grid cells where ice is never found (either because they are on land, or because they are too warm) to zero.
This is controlled by `mask_type`, set in the `decoder` section of the model config (at the top level for `ddpm`, which has no separate decoder):

```yaml
decoder:
  mask_type: active # active (active+land) | land (land only) | none to disable
```

`active` masks land and cells where sea ice has never been observed; `land` masks only land; `none` (or omitting `mask_type`, or `null`) disables masking entirely. Unlike `restrict_range` below, an unrecognised `mask_type` raises a `ValueError` rather than silently disabling masking. You don't need to point at mask files yourself: they're generated automatically by `datasets create` (currently for SSMIS datasets) and located from `base_path`, so requesting `active`/`land` for a target dataset without generated masks fails loudly at model build with a `FileNotFoundError`.

The models can also restrict the output data to the range 0-1, controlled by `restrict_range` in the same `decoder` section:

```yaml
decoder:
  restrict_range: sigmoid # none/sigmoid/clamp/tanh
```

By default, `restrict_range: sigmoid` is used, however `clamp`, `tanh` or `none` are also available.

## Checkpoints

A checkpoint is saved after each epoch to:

```
${BASE_DIR}/training/wandb/run-<date>-<id>/checkpoints/
```

where `BASE_DIR` is the `base_path` defined in your local config.
Pass this path to `evaluate` to assess the trained model.

## 4. Check results in W&B

Once training starts, the run appears in the W&B project `train` under the `turing-seaice` entity at [wandb.ai](https://wandb.ai).

### What to look for

| Panel | What it shows |
|-------|---------------|
| `train_loss` / `validation_loss` | Loss per epoch — watch for validation loss converging or diverging from training loss. |
| `train_{metric}_mean` / `validation_{metric}_mean` | Per-epoch mean of each metric (e.g. SIE error, IceNet accuracy). |
| `{metric}_per_forecast_day` | Metric broken down by forecast lead day, logged at the end of the run. Useful for diagnosing where skill drops at longer horizons. |

### Comparing runs

Use the W&B workspace to overlay multiple runs on the same chart, or group runs by config keys (e.g. `model`, `data`) to understand what drives performance differences across experiments.

## Multistage training

For `EncodeProcessDecode` models, components can be pretrained in isolation before a final finetuning step.
See [Run multistage training](train-multistage.md) for details.
