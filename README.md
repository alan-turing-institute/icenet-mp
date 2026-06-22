# IceNet Multimodal Pipeline

IceNet-MP is a multimodal pipeline for predicting sea ice.

## Setting up your environment

See the [Installation](https://alan-turing-institute.github.io/icenet-mp/user-guide/installation/) page in the docs for full installation instructions, including HPC-specific prerequisites.

See the [Configuration](https://alan-turing-institute.github.io/icenet-mp/user-guide/configuration/) page in the docs for details on local config files, HPC configs, and custom datasets.

## Running IceNet-MP commands

:information_source: Note that if you are running the below commands locally, specify the base path in your local config, then add the argument `--config-name <your local YAML config>`.

### Create

You will need a [CDS account](https://cds.climate.copernicus.eu/how-to-api) to download data with `anemoi` (e.g. the ERA5 data).

Run `uv run imp datasets create` to download datasets.

We make use of the fact that Anemoi datasets keep track of which groups of dates have been loaded to ensure that an interrupted download can be resumed simply by rerunning the `datasets create` command.

### Inspect

Run `uv run imp datasets inspect` to inspect datasets (i.e. to get dataset properties and statistical summaries of the variables).

### Train

You will need a [Weights & Biases account](https://docs.wandb.ai/models/quickstart) to run a training run.
[Generate an API key](https://docs.wandb.ai/models/quickstart) then run the following to allow automatic authentication.

```
export WANDB_API_KEY=<your_api_key>
wandb login
```

Run `uv run imp train` to train using the datasets specified in the config.

:information_source: This will save checkpoints to `${BASE_DIR}/training/wandb/run-${DATE}$-${RANDOM_STRING}/checkpoints/${CHECKPOINT_NAME}$.ckpt`. Where the `BASE_DIR` is the base path to the data defined in your config file.

:warning: If you are running on macOS, you may need to prepend your `uv` run command with `PYTORCH_ENABLE_MPS_FALLBACK=1`. For example:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run imp train
```

### Evaluate

Run `uv run imp evaluate --checkpoint PATH_TO_A_CHECKPOINT` to evaluate using a checkpoint from a training run.

### Visualisations

You can plot static images or animations of the raw data by adding the following option to your local config:
```
evaluate:
  callbacks:
    plotting:
      make_input_plots: true
```

Settings (output directories, styling, animation parameters) are read from `config.evaluate.callbacks.raw_inputs` in your YAML config files. Command-line options can override config values if needed.

## Adding a new model

### Background

An IceNet-MP model needs to be able to run over multiple different datasets with different dimensions.
These are structured in `NTCHW` format, where:
- `N` is the batch size,
- `T` is the number of history (forecast) steps for inputs (outputs)
- `C` is the number of channels or variables
- `H` is a height dimension
- `W` is a width dimension

`N` and `T` will be the same for all inputs, but `C`, `H` and `W` might vary.

Taking as an example, a batch size (`N=2`), 3 history steps and 4 forecast steps, we will have `k` inputs of shape `(2, 3, C_k, H_k, W_k)` and one output of shape `(2, 4, C_out, H_out, W_out)`.

### Standalone models

A standalone model will need to accept a `dict[str, TensorNTCHW]` which maps dataset names to an `NTCHW` Tensor of values.
The model might want to use one or more of these for training, and will need to produce an output with shape `N, T, C_out, H_out, W_out`.

As can be seen in the example below, a separate instance of the model is likely to be needed for each output to be predicted.

![image](docs/assets/pipeline-standalone.png)

Pros:
- all input variables are available without transformation

Cons:
- hard to add new inputs
- hard to add new outputs

### Processor models

A processor model is part of a larger encode-process-decode step.
Start by defining a latent space as `(C_latent, H_latent, W_latent)` - in the example below, this has been set to `(10, 64, 64)`.
The encode-process-decode model automatically creates one encoder for each input and one decoder for each output.
The dataset-specific encoder takes the input data and converts it to shape `(N, T, C_latent, H_latent, W_latent)`.
The `k` encoded datasets can then be combined in latent space to give a single dataset of shape `(N, T, k * C_latent, H_latent, W_latent)`.

This is then passed to the processor, which must accept input of shape `(N, T, k * C_latent, H_latent, W_latent)` and produce output of the same shape.

This output is then passed to one or more output-specific decoders which take input of shape `(N, T, k * C_latent, H_latent, W_latent)` and produce output of shape `(N, T, C_out, H_out, W_out)`.

![image](docs/assets/pipeline-encode-process-decode.png)

Pros:
- easy to add new inputs
- easy to add new outputs

Cons:
- input variables have been transformed into latent space

## Jupyter notebooks

There are various demonstrator Jupyter notebooks in the `notebooks` folder.
You can run these with `uv run --group notebooks jupyter notebook`.

A good one to start with is `notebooks/demo_pipeline.ipynb` which gives a more detailed overview of the pipeline.
