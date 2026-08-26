from unittest.mock import MagicMock

import numpy as np
import torch
from matplotlib.colors import Normalize

from icenet_mp.types import (
    AnemoiCleanupArgs,
    AnemoiDatasetStatus,
    AnemoiFinaliseArgs,
    AnemoiInitArgs,
    AnemoiInspectArgs,
    AnemoiLoadArgs,
    DiffColourmapSpec,
    Metadata,
    ProcessorOutput,
    UncertaintyArrays,
)


def test_anemoi_dataset_status_is_tuple_compatible() -> None:
    """Preserve tuple compatibility for Anemoi dataset status values."""
    status = AnemoiDatasetStatus(
        copy_in_progress=False,
        download_complete=True,
        is_finalised=True,
    )

    assert tuple(status) == (False, True, True)
    assert status.download_complete is True


def test_anemoi_command_args_keep_expected_defaults() -> None:
    """Keep expected defaults across Anemoi command argument dataclasses."""
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


def test_anemoi_inspect_args_preserve_requested_flags() -> None:
    """Preserve explicit inspect options without hidden defaults."""
    inspect = AnemoiInspectArgs(
        detailed=True,
        path="dataset.zarr",
        progress=False,
        size=True,
        statistics=False,
    )

    assert inspect.path == "dataset.zarr"
    assert inspect.detailed is True
    assert inspect.progress is False
    assert inspect.size is True
    assert inspect.statistics is False


def test_diff_colourmap_spec_preserves_normalisation_and_bounds() -> None:
    """Preserve normalisation, bounds and colourmap configuration."""
    norm = Normalize(vmin=-1.0, vmax=1.0)

    spec = DiffColourmapSpec(norm=norm, vmin=None, vmax=None, cmap="coolwarm")

    assert spec.norm is norm
    assert spec.vmin is None
    assert spec.vmax is None
    assert spec.cmap == "coolwarm"


def test_metadata_defaults_are_independent_and_optional() -> None:
    """Keep Metadata defaults optional and independent across instances."""
    first = Metadata()
    second = Metadata()

    first.vars_by_source = {"era5": ["2t"]}

    assert first.model is None
    assert first.n_points is None
    assert second.vars_by_source is None


def test_metadata_accepts_training_summary_fields() -> None:
    """Accept and preserve training-summary metadata fields."""
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
    """Default ProcessorOutput loss to None when omitted."""
    prediction = torch.randn(2, 3, 4, 5, 6)

    output = ProcessorOutput(prediction=prediction)

    assert output.prediction is prediction
    assert output.loss is None


def test_processor_output_keeps_custom_loss_tensor() -> None:
    """Preserve an explicitly supplied ProcessorOutput loss tensor."""
    prediction = torch.randn(1, 2, 3, 4, 5)
    loss = torch.tensor(0.25)

    output = ProcessorOutput(prediction=prediction, loss=loss)

    assert output.loss is loss


def test_uncertainty_arrays_preserve_named_tuple_fields() -> None:
    """Preserve array identities and tuple ordering for uncertainty values."""
    ground_truth = np.zeros((2, 3), dtype=np.float32)
    prediction = np.ones((2, 3), dtype=np.float32)
    uncertainty = np.full((2, 3), 0.1, dtype=np.float32)

    arrays = UncertaintyArrays(
        ground_truth=ground_truth,
        prediction=prediction,
        uncertainty=uncertainty,
    )

    assert arrays.ground_truth is ground_truth
    assert arrays.prediction is prediction
    assert arrays.uncertainty is uncertainty
    assert arrays[0] is ground_truth
    assert arrays[1] is prediction
    assert arrays[2] is uncertainty
