from dataclasses import replace
from datetime import date
from typing import Literal

import numpy as np
import pytest

from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.plotting_core import (
    compute_display_ranges,
    make_diff_colourmap,
)
from icenet_mp.visualisations.range_check import compute_range_check_report


def make_bad_prediction(
    base: np.ndarray,
    *,
    scale: float | None = None,
    outlier: float | None = None,
    fraction: float = 0.0,
    noise: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Helper to create mutated predictions for testing range check behavior."""
    prediction = base.copy()
    if scale is not None:
        prediction *= float(scale)
    if noise is not None:
        prediction += (rng or np.random.default_rng(0)).normal(
            0.0, float(noise), size=prediction.shape
        )
    if outlier is not None and fraction > 0.0:
        n = int(max(1, round(fraction * prediction.size)))
        prediction.ravel()[:n] = outlier
    return prediction


@pytest.mark.parametrize(
    "colourbar_strategy", ["shared", "separate"], ids=lambda s: f"cbar-{s}"
)
def test_sic_display_range_respects_strategy(
    sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
    colourbar_strategy: Literal["shared", "separate"],
) -> None:
    """Test that the display range respects the colourbar strategy."""
    ground_truth, prediction, _ = sic_pair_2d
    plot_spec = replace(DEFAULT_SIC_SPEC, colourbar_strategy=colourbar_strategy)
    (gt_vmin, gt_vmax), (p_vmin, p_vmax) = compute_display_ranges(
        ground_truth, prediction, plot_spec
    )
    if colourbar_strategy == "shared":
        assert (gt_vmin, gt_vmax) == (0.0, 1.0), (
            "Shared colourbar strategy should have vmin=0.0 and vmax=1.0 for Ground Truth"
        )
        assert (p_vmin, p_vmax) == (0.0, 1.0), (
            "Shared colourbar strategy should have vmin=0.0 and vmax=1.0 for Prediction"
        )
    else:
        assert gt_vmin <= gt_vmax, (
            "Separate colourbar strategy should have vmin<=vmax for Ground Truth"
        )
        assert p_vmin <= p_vmax, (
            "Separate colourbar strategy should have vmin<=vmax for Prediction"
        )


@pytest.mark.parametrize(
    "diff_mode", ["signed", "absolute", "smape"], ids=lambda s: f"diff-{s}"
)
def test_difference_colour_scale_modes(
    sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
    diff_mode: Literal["signed", "absolute", "smape"],
) -> None:
    """Test that the difference colour scale mode is correct."""
    ground_truth, prediction, _ = sic_pair_2d
    plot_spec = make_diff_colourmap(ground_truth - prediction, mode=diff_mode)
    if diff_mode == "signed":
        assert plot_spec.norm is not None, "Signed difference mode should have a norm"
        assert getattr(plot_spec.norm, "vcenter", None) == 0.0, (
            "Signed difference mode should have vcenter=0.0"
        )
    else:
        assert plot_spec.norm is None, "Difference mode should have a norm"
        assert plot_spec.vmin == 0.0, "Difference mode should have vmin=0.0"
        assert (plot_spec.vmax or 0) >= 0.0, "Difference mode should have vmax>=0.0"


# --- Range Check Tests with mutated predictions ---


@pytest.mark.parametrize(
    ("mutation_kwargs", "expect_colour", "expect_magnitude"),
    [
        # display clipping only: some values > 1.0 but scale similar
        ({"outlier": 1.2, "fraction": 0.25}, True, False),
        # magnitude mismatch: prediction 100x larger - will also trigger clipping since values > 1.0
        ({"scale": 100.0}, True, True),
        # both clipping and magnitude mismatch
        ({"scale": 100.0, "outlier": 1.2, "fraction": 0.10}, True, True),
        # no mutation -> no warnings
        ({}, False, False),
        # small noise can push values outside [0,1] range, triggering clipping warnings
        ({"noise": 0.01}, True, False),
    ],
    ids=[
        "display-clipping",
        "magnitude-mismatch",
        "both-issues",
        "no-issues",
        "noise-only",
    ],
)
def test_range_check_parametrized(
    sic_pair_2d: tuple[np.ndarray, np.ndarray, object],
    mutation_kwargs: dict,
    *,
    expect_colour: bool,
    expect_magnitude: bool,
) -> None:
    """Parametrised checks of compute_range_check_report using readable mutation helpers."""
    gt_base, pred_base, _ = sic_pair_2d
    pred = make_bad_prediction(pred_base, **mutation_kwargs)

    # Use shared display range for clarity in these tests
    spec_vmin, spec_vmax = 0.0, 1.0

    report = compute_range_check_report(
        gt_base,
        pred,
        vmin=spec_vmin,
        vmax=spec_vmax,
        outside_warn=DEFAULT_SIC_SPEC.outside_warn,
        severe_outside=DEFAULT_SIC_SPEC.severe_outside,
        include_shared_range_mismatch_check=DEFAULT_SIC_SPEC.include_shared_range_mismatch_check,
    )

    has_colour = any(
        "COLOUR ISSUE" in w or w.startswith("COLOUR ISSUE") for w in report.warnings
    )
    has_magnitude = any(
        "MAGNITUDE" in w or w.startswith("MAGNITUDE") for w in report.warnings
    )

    assert has_colour is expect_colour, (
        f"COLOUR expectation mismatch. got={report.warnings}"
    )
    assert has_magnitude is expect_magnitude, (
        f"MAGNITUDE expectation mismatch. got={report.warnings}"
    )


@pytest.mark.parametrize(
    ("outside_warn", "expected_warning"),
    [(0.01, True), (0.5, False)],
    ids=["low_threshold", "high_threshold"],
)
def test_outside_warn_threshold_behavior(
    outside_warn: float,
    *,
    expected_warning: bool,
) -> None:
    """Test that outside_warn threshold controls warning behavior."""
    # Create data with ~15% of values outside [0,1] - below severe threshold
    ground_truth = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
    prediction = np.array(
        [[0.5, 1.2, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32
    )  # 1/6 ≈ 17% > 1.0

    report = compute_range_check_report(
        ground_truth,
        prediction,
        vmin=0.0,
        vmax=1.0,
        outside_warn=outside_warn,
        severe_outside=0.20,  # 17% < 20% so won't trigger severe warning
        include_shared_range_mismatch_check=True,
    )

    has_warning = any(
        "COLOUR ISSUE" in w or w.startswith("COLOUR ISSUE") for w in report.warnings
    )
    assert has_warning == expected_warning, (
        f"Expected warning={expected_warning}, got warnings: {report.warnings}"
    )


def test_prefix_deduplication() -> None:
    """Test that category prefixes are deduplicated correctly."""
    # Create data that triggers multiple warnings of the same type
    ground_truth = np.array([[0.5]], dtype=np.float32)
    prediction = np.array([[50.0]], dtype=np.float32)  # magnitude mismatch

    report = compute_range_check_report(
        ground_truth,
        prediction,
        vmin=0.0,
        vmax=1.0,
        outside_warn=0.05,
        severe_outside=0.20,
        include_shared_range_mismatch_check=True,
    )

    # Should have warnings but with deduplicated prefixes
    assert len(report.warnings) > 0, "Expected some warnings"

    # Count prefixes - should have at most one of each type
    magnitude_prefixes = sum(1 for w in report.warnings if w.startswith("MAGNITUDE:"))
    colour_prefixes = sum(1 for w in report.warnings if w.startswith("COLOUR ISSUE:"))

    assert magnitude_prefixes <= 1, f"Too many MAGNITUDE: prefixes: {report.warnings}"
    assert colour_prefixes <= 1, f"Too many COLOUR ISSUE: prefixes: {report.warnings}"
