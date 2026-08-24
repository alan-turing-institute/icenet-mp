from unittest.mock import MagicMock

import torch
from matplotlib.colors import Normalize

from icenet_mp.types import (
    AnemoiCleanupArgs,
    AnemoiDatasetStatus,
    AnemoiFinaliseArgs,
    AnemoiInitArgs,
    AnemoiLoadArgs,
    DiffColourmapSpec,
    Metadata,
    ProcessorOutput,
)


def test_anemoi_dataset_status_is_tuple_compatible() -> None:
    status = AnemoiDatasetStatus(
        copy_in_progress=False,
        download_complete=True,
        is_finalised=True,
    )

    assert tuple(status) == (False, True, True)
    assert status.download_complete is True


def test_anemoi_command_args_keep_expected_defaults() -> None:
    recipe = MagicMock()

    cleanup = AnemoiCleanupArgs(path="dataset.zarr")
    initialise = AnemoiInitArgs(path="dataset.zarr", recipe=recipe)
    load = AnemoiLoadArgs(path="dataset.zarr", recipe=recipe)
    finalise = AnemoiFinaliseArgs(path="dataset.zarr", recipe=recipe)

    assert cleanup.command == "unused"
    assert cleanup.delta is None
    assert initialise.command == "unused"
    assert initialise.overwrite is False
    assert initialise.recipe is recipe
    assert load.command == "unused"
    assert load.recipe is recipe
    assert finalise.command == "unused"
    assert finalise.recipe is recipe


def test_diff_colourmap_spec_preserves_normalisation_and_bounds() -> None:
    norm = Normalize(vmin=-1.0, vmax=1.0)

    spec = DiffColourmapSpec(norm=norm, vmin=None, vmax=None, cmap="coolwarm")

    assert spec.norm is norm
    assert spec.vmin is None
    assert spec.vmax is None
    assert spec.cmap == "coolwarm"


def test_metadata_defaults_are_independent_and_optional() -> None:
    first = Metadata()
    second = Metadata()

    first.vars_by_source = {"era5": ["2t"]}

    assert first.model is None
    assert first.n_points is None
    assert second.vars_by_source is None


def test_metadata_accepts_training_summary_fields() -> None:
    metadata = Metadata(
        model="cnn-vit-cnn",
        max_epochs=20,
        current_epoch=7,
        start="2017-01-01",
        end="2019-12-31",
        cadence="24h",
        n_points=1095,
        n_history_steps=3,
        vars_by_source={"sic-ssmis": ["ice_conc"]},
    )

    assert metadata.model == "cnn-vit-cnn"
    assert metadata.current_epoch == 7
    assert metadata.n_history_steps == 3
    assert metadata.vars_by_source == {"sic-ssmis": ["ice_conc"]}


def test_processor_output_defaults_loss_to_none() -> None:
    prediction = torch.randn(2, 3, 4, 5, 6)

    output = ProcessorOutput(prediction=prediction)

    assert output.prediction is prediction
    assert output.loss is None


def test_processor_output_keeps_custom_loss_tensor() -> None:
    prediction = torch.randn(1, 2, 3, 4, 5)
    loss = torch.tensor(0.25)

    output = ProcessorOutput(prediction=prediction, loss=loss)

    assert output.loss is loss
