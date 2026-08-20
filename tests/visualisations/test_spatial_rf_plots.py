"""Unit tests for the opt-in spatial RF plotting helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from icenet_mp.input_explainability.spatial_rf import (
    LeadResult,
    SpatialRFResult,
)
from icenet_mp.visualisations.spatial_rf_plots import (
    plot_confirmation_refits,
    plot_lead_importance_heatmap,
    plot_model_quality,
    plot_reliability_grid,
    plot_spatial_rf_results,
    plot_stratum_importance,
    plot_target_mode_overlap,
)

if TYPE_CHECKING:
    from typing import Any


def _lead(
    lead: int,
    *,
    with_importance: bool = True,
    with_confirmation: bool = True,
    with_by_stratum: bool = True,
) -> LeadResult:
    """Construct a minimal LeadResult covering every optional plotting path."""
    importance: dict[str, dict[str, Any]] | None
    if not with_importance:
        importance = None
    else:
        importance = {
            "sic-ssmis/ice_conc": {
                "mean_mse_increase": 0.01,
                "std_mse_increase": 0.002,
                "fold_values": [0.011, 0.010, 0.009, 0.012, 0.009],
                "positive_folds": 5,
                "positive_fold_fraction": 1.0,
                "rank_stability": 0.9,
                "reliability": "stable",
                "by_stratum": (
                    {
                        "open_water": {
                            "mean_mse_increase": 0.005,
                            "std_mse_increase": 0.001,
                            "fold_values": [0.005] * 5,
                            "positive_folds": 5,
                            "positive_fold_fraction": 1.0,
                            "rank_stability": 1.0,
                            "reliability": "stable",
                        },
                        "pack_ice": {
                            "mean_mse_increase": 0.008,
                            "std_mse_increase": 0.001,
                            "fold_values": [0.008] * 5,
                            "positive_folds": 5,
                            "positive_fold_fraction": 1.0,
                            "rank_stability": 1.0,
                            "reliability": "stable",
                        },
                        "marginal_ice": {
                            "mean_mse_increase": 0.02,
                            "std_mse_increase": 0.003,
                            "fold_values": [0.02] * 5,
                            "positive_folds": 5,
                            "positive_fold_fraction": 1.0,
                            "rank_stability": 1.0,
                            "reliability": "stable",
                        },
                    }
                    if with_by_stratum
                    else {}
                ),
            },
            "era5/z_500": {
                "mean_mse_increase": 0.004,
                "std_mse_increase": 0.001,
                "fold_values": [0.004] * 5,
                "positive_folds": 5,
                "positive_fold_fraction": 1.0,
                "rank_stability": 0.7,
                "reliability": "stable",
                "add_to_sic_gain": 0.001 if with_confirmation else None,
                "drop_from_full_loss": -0.0005 if with_confirmation else None,
                "by_stratum": (
                    {
                        "open_water": {
                            "mean_mse_increase": 0.001,
                            "std_mse_increase": 0.0005,
                            "fold_values": [0.001] * 5,
                            "positive_folds": 4,
                            "positive_fold_fraction": 0.8,
                            "rank_stability": 1.0,
                            "reliability": "stable",
                        },
                        "pack_ice": {
                            "mean_mse_increase": 0.003,
                            "std_mse_increase": 0.001,
                            "fold_values": [0.003] * 5,
                            "positive_folds": 5,
                            "positive_fold_fraction": 1.0,
                            "rank_stability": 1.0,
                            "reliability": "stable",
                        },
                        "marginal_ice": {
                            "mean_mse_increase": 0.006,
                            "std_mse_increase": 0.001,
                            "fold_values": [0.006] * 5,
                            "positive_folds": 5,
                            "positive_fold_fraction": 1.0,
                            "rank_stability": 1.0,
                            "reliability": "stable",
                        },
                    }
                    if with_by_stratum
                    else {}
                ),
            },
        }
    return LeadResult(
        lead=lead,
        n_samples=120,
        mse=0.005,
        mae=0.05,
        baseline_mse=0.008,
        baseline_mae=0.06,
        sic_history_mse=0.006 if with_confirmation else None,
        by_stratum={},
        importance=importance,
        importance_interpretable=True,
        fold_boundaries=[],
    )


def _result(**kwargs: Any) -> SpatialRFResult:
    """Build a three-lead SpatialRFResult with sensible defaults."""
    leads = [_lead(1, **kwargs), _lead(2, **kwargs), _lead(3, **kwargs)]
    return SpatialRFResult(
        feature_names=["sic-ssmis/ice_conc_t-0", "era5/z_500_t-0"],
        leads=leads,
        metadata={
            "target_mode": "absolute",
            "rf_settings": {
                "backend": "hist_gradient_boosting",
                "importance_policy": "always",
            },
        },
    )


def test_importance_heatmap_writes_when_data_present(tmp_path: Path) -> None:
    """The heatmap is produced when at least one lead retains importance."""
    output = plot_lead_importance_heatmap(_result(), tmp_path)

    assert output is not None
    assert output.exists()
    assert output.name == "importance_heatmap.png"


def test_importance_heatmap_skips_when_no_importance(tmp_path: Path) -> None:
    """No importance data anywhere → no file written."""
    assert (
        plot_lead_importance_heatmap(_result(with_importance=False), tmp_path) is None
    )
    assert not (tmp_path / "importance_heatmap.png").exists()


def test_reliability_grid_writes_when_data_present(tmp_path: Path) -> None:
    """The reliability grid renders the three-level colourmap."""
    output = plot_reliability_grid(_result(), tmp_path)

    assert output is not None
    assert output.name == "reliability_grid.png"


def test_reliability_grid_skips_when_no_importance(tmp_path: Path) -> None:
    """Empty importance grid → no PNG."""
    assert plot_reliability_grid(_result(with_importance=False), tmp_path) is None
    assert not (tmp_path / "reliability_grid.png").exists()


def test_confirmation_refits_writes_when_metrics_present(tmp_path: Path) -> None:
    """Confirmation metrics are required for the bar chart."""
    output = plot_confirmation_refits(_result(), tmp_path)

    assert output is not None
    assert output.name == "confirmation_refits.png"


def test_confirmation_refits_skips_without_metrics(tmp_path: Path) -> None:
    """When no group exposes add_to_sic_gain/drop_from_full_loss, skip silently."""
    assert plot_confirmation_refits(_result(with_confirmation=False), tmp_path) is None
    assert not (tmp_path / "confirmation_refits.png").exists()


def test_stratum_importance_writes_when_all_strata_present(tmp_path: Path) -> None:
    """Stratum heatmap needs every stratum to be present in the importance data."""
    output = plot_stratum_importance(_result(), tmp_path)

    assert output is not None
    assert output.name == "stratum_importance.png"


def test_stratum_importance_skips_when_a_stratum_missing(tmp_path: Path) -> None:
    """If any stratum is absent, the plot is skipped to avoid misleading axes."""
    assert plot_stratum_importance(_result(with_by_stratum=False), tmp_path) is None
    assert not (tmp_path / "stratum_importance.png").exists()


def test_target_mode_overlap_writes_with_two_results(tmp_path: Path) -> None:
    """Two valid runs produce the scatter plot."""
    output = plot_target_mode_overlap(_result(), _result(), tmp_path)

    assert output is not None
    assert output.name == "target_mode_overlap.png"


def test_target_mode_overlap_skips_when_a_run_is_empty(tmp_path: Path) -> None:
    """Either side missing importance → no PNG."""
    assert (
        plot_target_mode_overlap(_result(with_importance=False), _result(), tmp_path)
        is None
    )
    assert not (tmp_path / "target_mode_overlap.png").exists()


def test_model_quality_writes_with_leads(tmp_path: Path) -> None:
    """Model quality plot needs at least one lead with mse/baseline_mse."""
    output = plot_model_quality(_result(), tmp_path)

    assert output is not None
    assert output.name == "model_quality.png"


def test_model_quality_skips_with_no_leads(tmp_path: Path) -> None:
    """No leads → no PNG."""
    empty = SpatialRFResult(feature_names=[], leads=[], metadata={"rf_settings": {}})

    assert plot_model_quality(empty, tmp_path) is None
    assert not (tmp_path / "model_quality.png").exists()


def test_orchestrator_returns_mapping_with_overlap(tmp_path: Path) -> None:
    """The orchestrator aggregates every plot and reports overlap only when asked."""
    paths = plot_spatial_rf_results(_result(), tmp_path, other=_result())

    assert set(paths) == {
        "importance_heatmap",
        "reliability_grid",
        "confirmation_refits",
        "stratum_importance",
        "model_quality",
        "target_mode_overlap",
    }
    for name, path in paths.items():
        assert path is not None, f"Expected plot {name} to be produced."
        assert path.exists()


def test_orchestrator_skips_overlap_when_no_other(tmp_path: Path) -> None:
    """Without a second result, the overlap entry is None and no file is written."""
    paths = plot_spatial_rf_results(_result(), tmp_path)

    assert paths["target_mode_overlap"] is None
    assert not (tmp_path / "target_mode_overlap.png").exists()


@pytest.mark.parametrize(
    "plot_fn",
    [
        plot_lead_importance_heatmap,
        plot_reliability_grid,
        plot_confirmation_refits,
        plot_stratum_importance,
    ],
)
def test_each_plot_skips_when_data_missing(
    plot_fn: Callable[[SpatialRFResult, Path], Path | None],
    tmp_path: Path,
) -> None:
    """Every importance-dependent plot returns ``None`` when importance is absent.

    ``plot_model_quality`` reads ``lead.mse`` rather than importance, so its
    skip path is covered by :func:`test_model_quality_skips_with_no_leads`.
    """
    result = _result(with_importance=False)

    assert plot_fn(result, tmp_path) is None
