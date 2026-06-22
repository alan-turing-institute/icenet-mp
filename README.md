# IceNet Multimodal Pipeline

IceNet-MP is a multimodal pipeline for predicting sea ice.

## Setting up your environment

See the [Installation](https://alan-turing-institute.github.io/icenet-mp/user-guide/installation/) page in the docs for full installation instructions, including HPC-specific prerequisites.

See the [Configuration](https://alan-turing-institute.github.io/icenet-mp/user-guide/configuration/) page in the docs for details on local config files, HPC configs, and custom datasets.

## Running IceNet-MP commands

See the [Commands](https://alan-turing-institute.github.io/icenet-mp/user-guide/commands/) page in the docs for details on `datasets create`, `datasets inspect`, `train`, and `evaluate`.

## Adding a new model

See [Add a model](https://alan-turing-institute.github.io/icenet-mp/how-to/add-a-model/) in the docs for the tensor format and a comparison of standalone vs. processor model architectures.

## Jupyter notebooks

There are various demonstrator Jupyter notebooks in the `notebooks` folder.
You can run these with `uv run --group notebooks jupyter notebook`.

A good one to start with is `notebooks/demo_pipeline.ipynb` which gives a more detailed overview of the pipeline.
