import logging
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from icenet_mp.callbacks.plotting_callback import PlottingCallback
from icenet_mp.types import ModelStepOutput
from icenet_mp.visualisations import DEFAULT_SIC_SPEC, Plotter


def _dataset_with_uncertainty() -> tuple[MagicMock, MagicMock]:
    dataset = MagicMock()
    target = MagicMock()
    target.name = "target"
    target.variable_names = ["ice_conc"]
    target.statistics = {"minimum": [0.0], "maximum": [2.0]}
    dataset.target = target

    source = MagicMock()
    source.name = "target"
    source.variable_names = ["ice_conc", "total_standard_uncertainty"]
    uncertainty_ds = MagicMock()
    uncertainty_ds.get_tchw.return_value = np.array(
        [[[[0.1, 0.2], [0.3, 1.1]]]], dtype=np.float32
    )
    source.subset.return_value = uncertainty_ds
    dataset.inputs = [source]
    return dataset, uncertainty_ds


def test_load_target_uncertainties_scales_and_masks() -> None:
    """Scale source uncertainty to target space and mask invalid values."""
    dataset, _ = _dataset_with_uncertainty()

    result = PlottingCallback.load_target_uncertainties(
        dataset, [datetime(2026, 8, 21, tzinfo=UTC)]
    )

    assert set(result) == {0}
    np.testing.assert_allclose(
        result[0],
        np.array([[[0.05, 0.1], [0.15, np.nan]]]),
        equal_nan=True,
    )
    dataset.inputs[0].subset.assert_called_once_with(
        variables=["total_standard_uncertainty"], normalise=False
    )


def test_load_target_uncertainties_skips_missing_source() -> None:
    """Return no uncertainty when the matching target input is unavailable."""
    dataset, _ = _dataset_with_uncertainty()
    dataset.inputs[0].name = "other"

    result = PlottingCallback.load_target_uncertainties(
        dataset, [datetime(2026, 8, 21, tzinfo=UTC)]
    )

    assert result == {}
    dataset.inputs[0].subset.assert_not_called()


def test_load_target_uncertainties_handles_data_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Skip uncertainty plotting when the source read fails."""
    dataset, uncertainty_ds = _dataset_with_uncertainty()
    uncertainty_ds.get_tchw.side_effect = ValueError("missing uncertainty")

    with caplog.at_level(logging.WARNING):
        result = PlottingCallback.load_target_uncertainties(
            dataset, [datetime(2026, 8, 21, tzinfo=UTC)]
        )

    assert result == {}
    assert "Could not load target uncertainty" in caplog.text


def test_load_target_uncertainties_rejects_invalid_target_range(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Skip scaling when the target normalisation range is invalid."""
    dataset, _ = _dataset_with_uncertainty()
    dataset.target.statistics = {"minimum": [1.0], "maximum": [1.0]}

    with caplog.at_level(logging.WARNING):
        result = PlottingCallback.load_target_uncertainties(
            dataset, [datetime(2026, 8, 21, tzinfo=UTC)]
        )

    assert result == {}
    assert "Could not scale target uncertainty" in caplog.text


def test_log_static_outputs_includes_uncertainty_image() -> None:
    """Log the uncertainty image alongside the standard static prediction."""
    plotter = Plotter(replace(DEFAULT_SIC_SPEC, selected_timestep=0))
    image_logger = MagicMock()
    outputs = ModelStepOutput(
        prediction=torch.zeros(1, 1, 1, 2, 2),
        target=torch.ones(1, 1, 1, 2, 2),
        loss=torch.tensor(0.0),
    )
    uncertainty = np.full((1, 2, 2), 0.1, dtype=np.float32)

    with (
        patch(
            "icenet_mp.visualisations.plotter.plot_static_prediction",
            return_value={"prediction": [MagicMock()]},
        ),
        patch(
            "icenet_mp.visualisations.plotter.plot_static_uncertainty",
            return_value={"uncertainty": [MagicMock()]},
        ) as plot_uncertainty,
    ):
        plotter.log_static_outputs(
            outputs,
            [datetime(2026, 8, 21, tzinfo=UTC)],
            [image_logger],
            ["ice_conc"],
            uncertainties={0: uncertainty},
        )

    plot_uncertainty.assert_called_once()
    assert [call.kwargs["key"] for call in image_logger.log_image.call_args_list] == [
        "output_static/prediction",
        "output_static/uncertainty",
    ]
