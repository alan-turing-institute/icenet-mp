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

If you are working on Baskerville, DAWN, or Isambard-AI, the datasets are already available on shared storage. Add the matching `platform` override and skip ahead to [step 2](#2-create-a-local-config):

```bash
uv run imp train --config-name <your-name>.local platform=isambardai  # or baskerville or dawn
```

### Downloading data locally

See the [`datasets create` command reference](../user-guide/commands.md#datasets-create) for prerequisites and usage.

## 2. Create a local config

Create a file at `icenet_mp/config/<your-name>.local.yaml`. The base config you inherit from depends on where you are running.

### On Isambard-AI, Baskerville, or DAWN

Inherit from `base` as usual, and add the `platform` override at the command line ([step 3](#3-run-training)) rather than baking it into your local config — it already points to the shared data storage for that system:

```yaml
defaults:
  - base
  - _self_
```

Use `platform=isambardai`, `platform=baskerville`, or `platform=dawn` depending on which system you are on.

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
The downloaded config is fully resolved, so update its `base_path` key to point at the right location for your machine.

## 3. Run training

```bash
uv run imp train --config-name <your-name>.local
```

Add `platform=isambardai` (or `baskerville`/`dawn`) if you are on one of those shared HPC systems.

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
