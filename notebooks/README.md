# IceNet-MP notebooks

The notebooks in this directory are supplementary examples and research artifacts for IceNet-MP. They are not required to use the package.

## Setup

For the current IceNet-MP CLI demo, use the project environment:

```bash
uv sync --group notebooks
cd notebooks
uv run jupyter lab
```

Research/case-study notebooks can have additional data, credential, environment, or checkpoint prerequisites described below or in the notebooks themselves.

## Recommended order

| Notebook | Decision and purpose | Prerequisite |
| --- | --- | --- |
| [`demo_pipeline.ipynb`](demo_pipeline.ipynb) | **Maintain.** Primary educational IceNet-MP walkthrough covering account-free synthetic data, training/evaluation artifacts, model architecture and persistence, Hydra configuration, multimodality, and an optional real-data route. | Start here. The default route uses generated synthetic data and local-file logging; the optional `demo_notebook` route documents the CDS/W&B prerequisites for real data. |
| [`layer_diagnostics.ipynb`](layer_diagnostics.ipynb) | **Maintain for diagnostics.** Activation-capture investigation for the current UNet/`quick_test` model and multimodal real-data path. | Advanced/research use. Supply a compatible checkpoint and existing real datasets; the notebook uses local-file logging for evaluation. |
| [`ARGO_data.ipynb`](ARGO_data.ipynb) | **Retain as a research example.** Download, inspect and grid ARGO float observations for the non-gridded data path. | Independent of the demo pipeline; requires network access and its geospatial/data dependencies. |
| [`case_study_whale_corridors.ipynb`](case_study_whale_corridors.ipynb) | **Retain as a case-study artifact.** Produce whale-corridor and shipping visualisations. | Requires the case-study data/configuration and is not part of the default CLI walkthrough. |

## Optional synthetic non-gridded workflow

The following pair predates the use of ARGO floats but remains useful when a controlled synthetic non-gridded dataset is needed. Retain them as an optional two-step research workflow:

1. [`extract_anomalies.ipynb`](extract_anomalies.ipynb) creates gridded ERA5 pressure anomalies.
2. [`degrid_and_visualise.ipynb`](degrid_and_visualise.ipynb) samples those anomalies into synthetic station and buoy observations.

Use [`nongriddedenv.yaml`](nongriddedenv.yaml) for this standalone research workflow.

## Removed legacy notebooks

The three legacy IceNet modelling notebooks and the early standalone persistence prototype were removed because they use the separate `icenet` code path rather than the current IceNet-MP pipeline. Notebook 0 was adapted from the [Environmental Data Science book gallery](https://github.com/eds-book-gallery/67a1e320-7c47-4ea9-8df8-e868326bc90b/tree/main); notebooks 1 and 2 came from the [IceNet notebooks repository](https://github.com/icenet-ai/icenet-notebooks).

The removed Conda environment files are not referenced by any retained notebook. Current IceNet-MP notebooks should use the project dependency groups where possible.
