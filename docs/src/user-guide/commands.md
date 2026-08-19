# Commands

!!! note
    When running locally, set `base_path` in your local config and pass `--config-name NAME_OF_LOCAL_CONFIG` to each command.

## `datasets create`

You will need a [CDS account](https://cds.climate.copernicus.eu/how-to-api) to download ERA5 data with `anemoi`.
By default, a sample dataset will be downloaded which should be small enough to fit on your personal computer. The full datasets are available on Isambard.

```bash
uv run imp datasets create
```

Anemoi tracks which date groups have been downloaded, so an interrupted download can be resumed by simply rerunning this command.

To create the synthetic dataset, use:

```bash
uv run imp datasets create --config-name synthetic
```

## `datasets inspect`

```bash
uv run imp datasets inspect
```

Prints basic properties of each dataset.
With the `--verbose` option it will also print statistical summaries of the variables.

## `train`

Standard (non-synthetic) runs use [Weights & Biases](https://docs.wandb.ai/models/quickstart).
Generate an API key, then authenticate:

```bash
export WANDB_API_KEY=<your_api_key>
wandb login
```

Trains the model end-to-end:

```bash
uv run imp train
```

To keep a W&B run local while debugging, set the logger's `offline` option. Metrics,
figures, and other run data are written locally without being uploaded to W&B:

```bash
uv run imp train loggers.wandb.offline=true
```

The same override works during evaluation:

```bash
uv run imp evaluate --checkpoint PATH_TO_A_CHECKPOINT loggers.wandb.offline=true
```

Alternatively, W&B's `WANDB_MODE=offline` environment variable can be used with either
command.

Synthetic experiments do not use W&B. Use the synthetic configuration, which
saves metrics and plotting artefacts locally under `${BASE_DIR}/report`:

```bash
uv run imp train --config-name synthetic
```

For `EncodeProcessDecode` models, pass `--multistage` to train each component separately before finetuning.
See [Train in stages](../how-to/train-multistage.md) for a full walkthrough.

```bash
uv run imp train --multistage
```

Checkpoints are saved to `${BASE_DIR}/training/wandb/run-<date>-<id>/checkpoints/<name>.ckpt`, where `BASE_DIR` is the base path defined in your config.

??? warning "macOS: MPS fallback"
    You may need to set `PYTORCH_ENABLE_MPS_FALLBACK=1`:

    ```bash
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run imp train
    ```

## `evaluate`

```bash
uv run imp evaluate --checkpoint PATH_TO_A_CHECKPOINT
```

### Visualisations

To plot static images or animations of the raw input data, add the following to your local config:

```yaml
evaluate:
  callbacks:
    plotting:
      make_input_plots: true
```

Output directories, styling, and animation parameters can be altered by changing `config.evaluate.callbacks.plotting.plot_spec`.
Any of these can be overridden at the command line.
