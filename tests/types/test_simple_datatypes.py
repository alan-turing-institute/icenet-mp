from unittest.mock import MagicMock

import numpy as np
import torch
from anemoi.datasets.create.recipe import Recipe
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


class TestAnemoiCommandArgs:
    """Tests for the Anemoi CLI command argument dataclasses."""

    def test_cleanup_defaults(self) -> None:
        """Default AnemoiCleanupArgs command and delta when omitted."""
        args = AnemoiCleanupArgs(path="dataset.zarr")

        assert args.command == "unused"
        assert args.delta is None

    def test_finalise_defaults(self) -> None:
        """Default AnemoiFinaliseArgs command while preserving the recipe."""
        recipe = MagicMock(spec=Recipe)

        args = AnemoiFinaliseArgs(path="dataset.zarr", recipe=recipe)

        assert args.command == "unused"
        assert args.recipe is recipe

    def test_init_defaults(self) -> None:
        """Default AnemoiInitArgs command and overwrite while preserving the recipe."""
        recipe = MagicMock(spec=Recipe)

        args = AnemoiInitArgs(path="dataset.zarr", recipe=recipe)

        assert args.command == "unused"
        assert args.overwrite is False
        assert args.recipe is recipe

    def test_load_defaults(self) -> None:
        """Default AnemoiLoadArgs command while preserving the recipe."""
        recipe = MagicMock(spec=Recipe)

        args = AnemoiLoadArgs(path="dataset.zarr", recipe=recipe)

        assert args.command == "unused"
        assert args.recipe is recipe


class TestAnemoiDatasetStatus:
    """Tests for AnemoiDatasetStatus."""

    def test_is_tuple_compatible(self) -> None:
        """Preserve tuple compatibility for Anemoi dataset status values."""
        status = AnemoiDatasetStatus(
            copy_in_progress=False,
            download_complete=True,
            is_finalised=True,
        )

        assert tuple(status) == (False, True, True)
        assert status.download_complete is True


class TestAnemoiInspectArgs:
    """Tests for AnemoiInspectArgs."""

    def test_preserve_requested_flags(self) -> None:
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


class TestDiffColourmapSpec:
    """Tests for DiffColourmapSpec."""

    def test_preserves_normalisation_and_bounds(self) -> None:
        """Preserve normalisation, bounds and colourmap configuration."""
        norm = Normalize(vmin=-1.0, vmax=1.0)

        spec = DiffColourmapSpec(norm=norm, vmin=None, vmax=None, cmap="coolwarm")

        assert spec.norm is norm
        assert spec.vmin is None
        assert spec.vmax is None
        assert spec.cmap == "coolwarm"


class TestMetadata:
    """Tests for Metadata."""

    def test_accepts_training_summary_fields(self) -> None:
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

    def test_defaults_are_independent_and_optional(self) -> None:
        """Keep Metadata defaults optional and independent across instances."""
        first = Metadata()
        second = Metadata()

        first.vars_by_source = {"era5": ["2t"]}

        assert first.model is None
        assert first.n_points is None
        assert second.vars_by_source is None


class TestProcessorOutput:
    """Tests for ProcessorOutput."""

    def test_defaults_loss_to_none(self) -> None:
        """Default ProcessorOutput loss to None when omitted."""
        prediction = torch.randn(2, 3, 4, 5, 6)

        output = ProcessorOutput(prediction=prediction)

        assert output.prediction is prediction
        assert output.loss is None

    def test_keeps_custom_loss_tensor(self) -> None:
        """Preserve an explicitly supplied ProcessorOutput loss tensor."""
        prediction = torch.randn(1, 2, 3, 4, 5)
        loss = torch.tensor(0.25)

        output = ProcessorOutput(prediction=prediction, loss=loss)

        assert output.loss is loss


class TestUncertaintyArrays:
    """Tests for UncertaintyArrays."""

    def test_preserve_named_tuple_fields(self) -> None:
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
