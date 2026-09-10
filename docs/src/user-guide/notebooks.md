# Notebooks

IceNet-MP includes a small set of notebooks for worked examples, research workflows and diagnostics. The notebooks supplement the command and configuration documentation; the package does not depend on them.

## Setup

For the current IceNet-MP CLI demo, from the repository root:

```bash
uv sync --group notebooks
cd notebooks
uv run jupyter lab
```

Research and case-study notebooks can require additional datasets, credentials, environment dependencies, or an existing checkpoint. Check their prerequisites before running them.

## Where to start

Start with [`demo_pipeline.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/demo_pipeline.ipynb) for the primary educational IceNet-MP walkthrough. Its default route uses generated synthetic data and local-file logging, while a separate optional `demo_notebook` route explains the CDS and Weights & Biases prerequisites for the multimodal real-data pipeline.

The notebook maintenance decisions for #417 are:

- **Maintain** [`demo_pipeline.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/demo_pipeline.ipynb) as the primary educational CLI walkthrough, including end-to-end execution, artifacts, model architecture and persistence, configuration/Hydra, multimodality, and the optional real-data route.
- **Maintain for diagnostics** [`layer_diagnostics.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/layer_diagnostics.ipynb). It demonstrates activation capture for the current UNet/`quick_test` model and multimodal real-data path; it requires a compatible checkpoint and existing real datasets and is not part of the default runnable walkthrough.
- **Retain as a research example** [`ARGO_data.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/ARGO_data.ipynb) for downloading and gridding real non-gridded ARGO observations.
- **Retain as a case-study artifact** [`case_study_whale_corridors.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/case_study_whale_corridors.ipynb). It requires its supporting data/configuration and is not part of the default CLI path.

## Synthetic non-gridded data

Two retained research notebooks provide a controlled synthetic alternative to ARGO observations. Run them in this order:

1. [`extract_anomalies.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/extract_anomalies.ipynb)
2. [`degrid_and_visualise.ipynb`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/degrid_and_visualise.ipynb)

They use [`nongriddedenv.yaml`](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/nongriddedenv.yaml) and are retained as an optional research workflow.

The legacy IceNet model notebooks and early standalone persistence prototype were removed because they use the separate `icenet` code path rather than the current IceNet-MP pipeline. The removed Conda environment files are not referenced by retained notebooks.

For a concise inventory and maintenance notes, see the [`notebooks` README](https://github.com/alan-turing-institute/icenet-mp/blob/main/notebooks/README.md).
