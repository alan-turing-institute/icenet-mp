"""Unit tests for the opt-in spatial RF plotting CLI helper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

import icenet_mp.cli.plotting as plotting_mod
from icenet_mp.cli.plotting import maybe_plot_spatial_rf_results


def test_skips_silently_when_plot_results_flag_is_false(tmp_path: Path) -> None:
    """Default flag (false) → no plotting, no file."""
    config = OmegaConf.create({"rf": {"spatial": {"plot_results": False}}})
    result = MagicMock()

    maybe_plot_spatial_rf_results(config, result, tmp_path)

    assert not any(tmp_path.iterdir())


def test_skips_silently_when_flag_missing(tmp_path: Path) -> None:
    """Missing flag defaults to false; no plotting happens."""
    config = OmegaConf.create({"rf": {"spatial": {}}})
    result = MagicMock()

    maybe_plot_spatial_rf_results(config, result, tmp_path)

    assert not any(tmp_path.iterdir())


def test_swallows_plotting_failures(tmp_path: Path) -> None:
    """A plotting failure is logged but does not break the surrounding run."""
    config = OmegaConf.create({"rf": {"spatial": {"plot_results": True}}})
    result = MagicMock()

    with patch(
        "icenet_mp.visualisations.spatial_rf_plots.plot_spatial_rf_results",
        side_effect=RuntimeError("matplotlib is broken"),
    ):
        maybe_plot_spatial_rf_results(config, result, tmp_path)

    assert not any(tmp_path.iterdir())


def test_logs_skipped_plots_when_some_missing(tmp_path: Path) -> None:
    """Successful and skipped plot names are both surfaced in the logs."""
    config = OmegaConf.create({"rf": {"spatial": {"plot_results": True}}})
    result = MagicMock()
    mixed_paths = {
        "importance_heatmap": Path("x.png"),
        "reliability_grid": None,
        "metric_grid": None,
    }

    with (
        patch(
            "icenet_mp.visualisations.spatial_rf_plots.plot_spatial_rf_results",
            return_value=mixed_paths,
        ),
        patch.object(plotting_mod, "logger") as mock_logger,
    ):
        maybe_plot_spatial_rf_results(config, result, tmp_path)

    written_calls = [
        call
        for call in mock_logger.info.call_args_list
        if call.args and "Spatial RF plots written to" in call.args[0]
    ]
    skipped_calls = [
        call
        for call in mock_logger.info.call_args_list
        if call.args and "Spatial RF plots skipped:" in call.args[0]
    ]

    assert written_calls, "expected success log line to be emitted"
    assert skipped_calls, "expected skipped log line to be emitted"

    written_msg = written_calls[-1].args[2]
    skipped_msg = skipped_calls[-1].args[1]

    assert "importance_heatmap" in written_msg
    for name in ("reliability_grid", "metric_grid"):
        assert name in skipped_msg
    assert "importance_heatmap" not in skipped_msg
