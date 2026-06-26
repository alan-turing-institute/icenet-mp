# Train a model

Single-stage training trains the full model end-to-end in one pass.
It is the default and works for all model architectures.

```bash
uv run imp train
```

## Prerequisites

You will need a [Weights & Biases account](https://docs.wandb.ai/models/quickstart).
Generate an API key, then authenticate before running any training command:

```bash
export WANDB_API_KEY=<your_api_key>
wandb login
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

## Checkpoints

A checkpoint is saved after each epoch to:

```
${BASE_DIR}/training/wandb/run-<date>-<id>/checkpoints/
```

where `BASE_DIR` is the `base_path` defined in your local config.
Pass this path to `evaluate` to assess the trained model.

## Multistage training

For `EncodeProcessDecode` models, components can be pretrained in isolation before a final finetuning step.
See [Run multistage training](train-multistage.md) for details.
