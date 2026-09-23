"""Tests for icenet_mp/visualisations/difference_calculator.py.

Covers the difference and standardised-difference computations for a
ground-truth/prediction pair.
"""

import numpy as np
import pytest
from matplotlib.colors import TwoSlopeNorm

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import DiffMode
from icenet_mp.visualisations.difference_calculator import (
    DifferenceCalculator,
)


class TestComputeDifference:
    ground_truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    prediction = np.array([[0.5, 2.5], [2.0, 5.0]])

    def test_signed(self) -> None:
        """Signed difference is ground_truth - prediction."""
        result = DifferenceCalculator(DiffMode.SIGNED).difference(
            self.ground_truth, self.prediction
        )

        np.testing.assert_allclose(result, [[0.5, -0.5], [1.0, -1.0]])

    def test_absolute(self) -> None:
        """Absolute difference is |ground_truth - prediction|."""
        result = DifferenceCalculator(DiffMode.ABSOLUTE).difference(
            self.ground_truth, self.prediction
        )

        np.testing.assert_allclose(result, [[0.5, 0.5], [1.0, 1.0]])

    def test_smape(self) -> None:
        """SMAPE difference normalises the absolute error by the mean magnitude."""
        result = DifferenceCalculator(DiffMode.SMAPE).difference(
            self.ground_truth, self.prediction
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

        result = DifferenceCalculator(DiffMode.SMAPE).difference(
            ground_truth, prediction
        )

        assert np.isfinite(result).all()

    def test_rejects_shape_mismatch(self) -> None:
        """Reject ground truth/prediction arrays with mismatched shapes."""
        with pytest.raises(InvalidArrayError, match="matching shapes"):
            DifferenceCalculator(DiffMode.SIGNED).difference(
                self.ground_truth,
                np.zeros((3, 3)),
            )

    def test_invalid_mode_raises(self) -> None:
        """An unrecognised difference mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown difference mode: dummy"):
            DifferenceCalculator("dummy").difference(  # type: ignore[arg-type]
                self.ground_truth,
                self.prediction,
            )

    def test_uses_mode_bound_at_construction_when_not_overridden(self) -> None:
        """A diff_mode bound at construction is used when no per-call mode is given."""
        result = DifferenceCalculator(DiffMode.SIGNED).difference(
            self.ground_truth, self.prediction
        )

        np.testing.assert_allclose(result, [[0.5, -0.5], [1.0, -1.0]])


class TestComputeStandardisedDifference:
    def test_basic(self) -> None:
        """Compute signed prediction error in units of uncertainty."""
        ground_truth = np.array([[0.5, 0.8], [0.4, 0.2]], dtype=np.float32)
        prediction = np.array([[0.4, 0.6], [0.3, 0.5]], dtype=np.float32)
        uncertainty = np.array([[0.1, 0.2], [0.05, 0.1]], dtype=np.float32)

        result = DifferenceCalculator(DiffMode.SIGNED).standardised_difference(
            ground_truth, prediction, uncertainty
        )

        np.testing.assert_allclose(result, [[1.0, 1.0], [2.0, -3.0]])

    def test_masks_invalid_uncertainty(self) -> None:
        """Mask values where uncertainty is zero, negative or non-finite."""
        ground_truth = np.ones((2, 2), dtype=np.float32)
        prediction = np.zeros((2, 2), dtype=np.float32)
        uncertainty = np.array([[0.5, 0.0], [-1.0, np.nan]], dtype=np.float32)

        result = DifferenceCalculator(DiffMode.SIGNED).standardised_difference(
            ground_truth, prediction, uncertainty
        )

        assert result[0, 0] == pytest.approx(2.0)
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 0])
        assert np.isnan(result[1, 1])

    def test_rejects_shape_mismatch(self) -> None:
        """Reject input arrays with mismatched shapes."""
        with pytest.raises(InvalidArrayError, match="matching shapes"):
            DifferenceCalculator(DiffMode.SIGNED).standardised_difference(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((3, 3), dtype=np.float32),
            )

    def test_rejects_1d_arrays(self) -> None:
        """Reject 1D ground truth/prediction/uncertainty arrays."""
        with pytest.raises(InvalidArrayError, match="Expected 2D"):
            DifferenceCalculator(DiffMode.SIGNED).standardised_difference(
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
            )

    def test_accepts_3d_video_arrays(self) -> None:
        """Compute a standardised difference over a [T, H, W] video stack."""
        ground_truth = np.ones((2, 2, 2), dtype=np.float32)
        prediction = np.zeros((2, 2, 2), dtype=np.float32)
        uncertainty = np.full((2, 2, 2), 0.5, dtype=np.float32)

        result = DifferenceCalculator(DiffMode.SIGNED).standardised_difference(
            ground_truth, prediction, uncertainty
        )

        assert result.shape == (2, 2, 2)
        np.testing.assert_allclose(result, 2.0)


class TestMakeDiffColourmap:
    def test_signed_scalar(self) -> None:
        """A scalar sample yields a symmetric TwoSlopeNorm around zero."""
        spec = DifferenceCalculator(DiffMode.SIGNED).colour_style(2.5)

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vcenter == pytest.approx(0.0)
        assert spec.norm.vmin == pytest.approx(-2.5)
        assert spec.norm.vmax == pytest.approx(2.5)
        assert spec.vmin is None
        assert spec.vmax is None
        assert spec.cmap == "RdBu_r"

    def test_signed_scalar_below_one_still_uses_unit_floor(self) -> None:
        """A small scalar sample still gets at least a +/-1 symmetric range."""
        spec = DifferenceCalculator(DiffMode.SIGNED).colour_style(0.1)

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vmin == pytest.approx(-1.0)
        assert spec.norm.vmax == pytest.approx(1.0)

    def test_signed_array(self) -> None:
        """An array sample uses the largest absolute extreme for a symmetric range."""
        sample = np.array([-2.0, 3.0, 0.5])

        spec = DifferenceCalculator(DiffMode.SIGNED).colour_style(sample)

        assert isinstance(spec.norm, TwoSlopeNorm)
        assert spec.norm.vmin == pytest.approx(-3.0)
        assert spec.norm.vmax == pytest.approx(3.0)
        assert spec.cmap == "RdBu_r"

    def test_absolute_scalar(self) -> None:
        """A scalar sample for absolute mode sets vmax directly."""
        spec = DifferenceCalculator(DiffMode.ABSOLUTE).colour_style(0.75)

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(0.75)
        assert spec.cmap == "magma"

    def test_absolute_array(self) -> None:
        """An array sample for absolute mode sets vmax from the array's max."""
        sample = np.array([0.1, 0.9, 0.4])

        spec = DifferenceCalculator(DiffMode.ABSOLUTE).colour_style(sample)

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(0.9)
        assert spec.cmap == "magma"

    def test_smape_scalar(self) -> None:
        """SMAPE mode behaves like absolute mode for a scalar sample."""
        spec = DifferenceCalculator(DiffMode.SMAPE).colour_style(1.5)

        assert spec.norm is None
        assert spec.vmin == pytest.approx(0.0)
        assert spec.vmax == pytest.approx(1.5)
        assert spec.cmap == "magma"

    def test_smape_array(self) -> None:
        """SMAPE mode behaves like absolute mode for an array sample."""
        sample = np.array([0.2, 0.6])

        spec = DifferenceCalculator(DiffMode.SMAPE).colour_style(sample)

        assert spec.vmax == pytest.approx(0.6)
        assert spec.cmap == "magma"

    def test_vmax_floor_avoids_zero_width_range(self) -> None:
        """A zero (or negative) sample still yields a strictly positive vmax."""
        spec = DifferenceCalculator(DiffMode.ABSOLUTE).colour_style(0.0)

        assert spec.vmax == pytest.approx(1e-6)

    def test_invalid_mode_raises(self) -> None:
        """An unrecognised mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown difference mode: dummy"):
            DifferenceCalculator("dummy").colour_style(1.0)  # type: ignore[arg-type]
