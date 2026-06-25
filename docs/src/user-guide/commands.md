# Commands

!!! note
    When running locally, set `base_path` in your local config and pass `--config-name NAME_OF_LOCAL_CONFIG` to each command.

## `datasets create`

You will need a [CDS account](https://cds.climate.copernicus.eu/how-to-api) to download ERA5 data with `anemoi`.

```bash
uv run imp datasets create
```

Anemoi tracks which date groups have been downloaded, so an interrupted download can be resumed by simply rerunning this command.

## `datasets inspect`

```bash
uv run imp datasets inspect
```

Prints basic properties of each dataset.
With the `--verbose` option it will also print statistical summaries of the variables.

## `train`

You will need a [Weights & Biases account](https://docs.wandb.ai/models/quickstart).
Generate an API key then authenticate:

```bash
export WANDB_API_KEY=<your_api_key>
wandb login
```

```bash
uv run imp train
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

Output directories, styling, and animation parameters are controlled by `config.evaluate.callbacks.raw_inputs`. Any of these can be overridden at the command line.
