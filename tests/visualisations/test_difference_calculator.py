"""Tests for icenet_mp/visualisations/difference_calculator.py.

Covers the difference and display-range computations for a ground-truth/prediction
pair. compute_standardised_difference is tested in tests/visualisations/test_uncertainty.py
alongside its sibling plot_static_uncertainty, not here.
"""

from dataclasses import replace

import numpy as np
import pytest
from matplotlib.colors import TwoSlopeNorm

from icenet_mp.types import PlotSpec
from icenet_mp.visualisations.difference_calculator import DifferenceCalculator


class TestComputeDifference:
    ground_truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    prediction = np.array([[0.5, 2.5], [2.0, 5.0]])

    def test_signed(self) -> None:
        """Signed difference is ground_truth - prediction."""
        result = DifferenceCalculator().compute_difference(
            self.ground_truth, self.prediction, "signed"
        )

        np.testing.assert_allclose(result, [[0.5, -0.5], [1.0, -1.0]])

    def test_absolute(self) -> None:
        """Absolute difference is |ground_truth - prediction|."""
        result = DifferenceCalculator().compute_difference(
            self.ground_truth, self.prediction, "absolute"
        )

        np.testing.assert_allclose(result, [[0.5, 0.5], [1.0, 1.0]])

    def test_smape(self) -> None:
        """SMAPE difference normalises the absolute error by the mean magnitude."""
        result = DifferenceCalculator().compute_difference(
            self.ground_truth, self.prediction, "smape"
        )

        expected = np.array(
            [
                [0.5 / 0.75, 0.5 / 2.25],
                [1.0 / 2.5, 1.0 / 4.5],
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_smape_avoids_division_by_zero(self) -> None:
        """A near-zero denominator is clipped rather than dividing by zero."""
        ground_truth = np.array([0.0])
        prediction = np.array([0.0])

        result = DifferenceCalculator().compute_difference(
            ground_truth, prediction, "smape"
        )

        assert np.isfinite(result).all()

    def test_invalid_mode_raises(self) -> None:
        """An unrecognised difference mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid difference mode"):
            DifferenceCalculator().compute_difference(
                self.ground_truth,
                self.prediction,
                "bogus",  # type: ignore[arg-type]
            )


class TestMakeDiffColourmap:
    def test_signed_scalar(self) -> None:
        """A scalar sample yields a symmetric TwoSlopeNorm around zero."""
        spec = DifferenceCalculator().make_diff_colourmap(2.5, mode="signed")

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vcenter == pytest.approx(0.0)
        assert spec.norm.vmin == pytest.approx(-2.5)
        assert spec.norm.vmax == pytest.approx(2.5)
        assert spec.vmin is None
        assert spec.vmax is None
        assert spec.cmap == "RdBu_r"

    def test_signed_scalar_below_one_still_uses_unit_floor(self) -> None:
        """A small scalar sample still gets at least a +/-1 symmetric range."""
        spec = DifferenceCalculator().make_diff_colourmap(0.1, mode="signed")

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vmin == pytest.approx(-1.0)
        assert spec.norm.vmax == pytest.approx(1.0)

    def test_signed_array(self) -> None:
        """An array sample uses the largest absolute extreme for a symmetric range."""
        sample = np.array([-2.0, 3.0, 0.5])

        spec = DifferenceCalculator().make_diff_colourmap(sample, mode="signed")

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vmin == pytest.approx(-3.0)
        assert spec.norm.vmax == pytest.approx(3.0)
        assert spec.cmap == "RdBu_r"

    def test_absolute_scalar(self) -> None:
        """A scalar sample for absolute mode sets vmax directly."""
        spec = DifferenceCalculator().make_diff_colourmap(0.75, mode="absolute")

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(0.75)
        assert spec.cmap == "magma"

    def test_absolute_array(self) -> None:
        """An array sample for absolute mode sets vmax from the array's max."""
        sample = np.array([0.1, 0.9, 0.4])

        spec = DifferenceCalculator().make_diff_colourmap(sample, mode="absolute")

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(0.9)
        assert spec.cmap == "magma"

    def test_smape_scalar(self) -> None:
        """SMAPE mode behaves like absolute mode for a scalar sample."""
        spec = DifferenceCalculator().make_diff_colourmap(1.5, mode="smape")

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(1.5)
        assert spec.cmap == "magma"

    def test_smape_array(self) -> None:
        """SMAPE mode behaves like absolute mode for an array sample."""
        sample = np.array([0.2, 0.6])

        spec = DifferenceCalculator().make_diff_colourmap(sample, mode="smape")

        assert spec.vmax == pytest.approx(0.6)
        assert spec.cmap == "magma"

    def test_vmax_floor_avoids_zero_width_range(self) -> None:
        """A zero (or negative) sample still yields a strictly positive vmax."""
        spec = DifferenceCalculator().make_diff_colourmap(0.0, mode="absolute")

        assert spec.vmax == pytest.approx(1e-6)

    def test_invalid_mode_raises(self) -> None:
        """An unrecognised mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown difference mode"):
            DifferenceCalculator().make_diff_colourmap(1.0, mode="bogus")  # type: ignore[arg-type]


class TestPrepareDifferenceStream:
    ground_truth_stream = np.array(
        [
            [[0.5, 0.2], [0.1, 0.9]],
            [[0.4, 0.3], [0.2, 0.6]],
            [[0.7, 0.1], [0.3, 0.5]],
        ]
    )
    prediction_stream = np.array(
        [
            [[0.4, 0.3], [0.2, 0.7]],
            [[0.5, 0.1], [0.3, 0.5]],
            [[0.6, 0.3], [0.1, 0.6]],
        ]
    )

    def test_include_difference_false_short_circuits(self) -> None:
        """include_difference=False returns (None, None) regardless of strategy."""
        difference_stream, colour_scale = (
            DifferenceCalculator().prepare_difference_stream(
                include_difference=False,
                diff_mode="signed",
                strategy="precompute",
                ground_truth_stream=self.ground_truth_stream,
                prediction_stream=self.prediction_stream,
            )
        )

        assert difference_stream is None
        assert colour_scale is None

    def test_precompute_strategy_returns_full_difference_stream(self) -> None:
        """Precompute returns the full elementwise difference stream and a colour scale."""
        difference_stream, colour_scale = (
            DifferenceCalculator().prepare_difference_stream(
                include_difference=True,
                diff_mode="signed",
                strategy="precompute",
                ground_truth_stream=self.ground_truth_stream,
                prediction_stream=self.prediction_stream,
            )
        )

        assert difference_stream is not None
        np.testing.assert_allclose(
            difference_stream, self.ground_truth_stream - self.prediction_stream
        )
        assert colour_scale is not None
        assert colour_scale.cmap == "RdBu_r"

    def test_two_pass_strategy_scans_for_colour_scale_only(self) -> None:
        """two-pass returns no difference stream but derives the colour scale."""
        difference_stream, colour_scale = (
            DifferenceCalculator().prepare_difference_stream(
                include_difference=True,
                diff_mode="absolute",
                strategy="two-pass",
                ground_truth_stream=self.ground_truth_stream,
                prediction_stream=self.prediction_stream,
            )
        )

        assert difference_stream is None
        assert colour_scale is not None
        expected_max = float(
            np.nanmax(np.abs(self.ground_truth_stream - self.prediction_stream))
        )
        assert colour_scale.vmax == pytest.approx(expected_max)
        assert colour_scale.vmin == pytest.approx(0.0)

    def test_two_pass_strategy_signed_takes_absolute_of_extremes(self) -> None:
        """two-pass with signed mode uses the largest absolute per-frame extreme."""
        _, colour_scale = DifferenceCalculator().prepare_difference_stream(
            include_difference=True,
            diff_mode="signed",
            strategy="two-pass",
            ground_truth_stream=self.ground_truth_stream,
            prediction_stream=self.prediction_stream,
        )

        assert colour_scale is not None
        assert isinstance(colour_scale.norm, TwoSlopeNorm)
        raw_max = float(
            np.nanmax(np.abs(self.ground_truth_stream - self.prediction_stream))
        )
        # make_diff_colourmap enforces a minimum symmetric span of +/-1.0.
        expected_max = max(1.0, raw_max)
        assert colour_scale.norm.vmax == pytest.approx(expected_max)
        assert colour_scale.norm.vmin == pytest.approx(-expected_max)

    def test_per_frame_strategy_returns_nothing_precomputed(self) -> None:
        """per-frame defers all computation to per-frame calls, returning (None, None)."""
        difference_stream, colour_scale = (
            DifferenceCalculator().prepare_difference_stream(
                include_difference=True,
                diff_mode="signed",
                strategy="per-frame",
                ground_truth_stream=self.ground_truth_stream,
                prediction_stream=self.prediction_stream,
            )
        )

        assert difference_stream is None
        assert colour_scale is None

    def test_invalid_strategy_raises(self) -> None:
        """An unrecognised strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown DiffStrategy"):
            DifferenceCalculator().prepare_difference_stream(
                include_difference=True,
                diff_mode="signed",
                strategy="bogus",  # type: ignore[arg-type]
                ground_truth_stream=self.ground_truth_stream,
                prediction_stream=self.prediction_stream,
            )


class TestComputeDisplayRanges:
    ground_truth = np.array([[0.1, 0.5], [0.9, 0.3]], dtype=np.float32)
    prediction = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)

    def test_shared_strategy_uses_ground_truth_range_for_both(self) -> None:
        """'shared' uses the ground-truth range for both ground truth and prediction."""
        plot_spec = replace(PlotSpec(), colourbar_strategy="shared")

        gt_range, pred_range = DifferenceCalculator().compute_display_ranges(
            self.ground_truth, self.prediction, plot_spec
        )

        assert gt_range == (pytest.approx(0.1), pytest.approx(0.9))
        assert pred_range == (pytest.approx(0.1), pytest.approx(0.9))

    def test_separate_strategy_uses_each_panels_own_range(self) -> None:
        """'separate' uses each panel's own data range."""
        plot_spec = replace(PlotSpec(), colourbar_strategy="separate")

        gt_range, pred_range = DifferenceCalculator().compute_display_ranges(
            self.ground_truth, self.prediction, plot_spec
        )

        assert gt_range == (pytest.approx(0.1), pytest.approx(0.9))
        assert pred_range == (pytest.approx(0.2), pytest.approx(0.8))
