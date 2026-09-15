"""Tests for icenet_mp/visualisations/difference_calculator.py.

Covers the difference and standardised-difference computations for a
ground-truth/prediction pair.
"""

import numpy as np
import pytest

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
            self.ground_truth, self.prediction, DiffMode.SIGNED
        )

        np.testing.assert_allclose(result, [[0.5, -0.5], [1.0, -1.0]])

    def test_absolute(self) -> None:
        """Absolute difference is |ground_truth - prediction|."""
        result = DifferenceCalculator(DiffMode.SIGNED).difference(
            self.ground_truth, self.prediction, DiffMode.ABSOLUTE
        )

        np.testing.assert_allclose(result, [[0.5, 0.5], [1.0, 1.0]])

    def test_smape(self) -> None:
        """SMAPE difference normalises the absolute error by the mean magnitude."""
        result = DifferenceCalculator(DiffMode.SIGNED).difference(
            self.ground_truth, self.prediction, DiffMode.SMAPE
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

        result = DifferenceCalculator(DiffMode.SIGNED).difference(
            ground_truth, prediction, DiffMode.SMAPE
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
        with pytest.raises(ValueError, match="Invalid difference mode"):
            DifferenceCalculator(DiffMode.SIGNED).difference(
                self.ground_truth,
                self.prediction,
                "bogus",  # type: ignore[arg-type]
            )

    def test_uses_mode_bound_at_construction_when_not_overridden(self) -> None:
        """A diff_mode bound at construction is used when no per-call mode is given."""
        result = DifferenceCalculator(DiffMode.SIGNED).difference(
            self.ground_truth, self.prediction
        )

        np.testing.assert_allclose(result, [[0.5, -0.5], [1.0, -1.0]])

    def test_per_call_mode_overrides_construction_default(self) -> None:
        """An explicit per-call diff_mode takes precedence over the bound default."""
        result = DifferenceCalculator(DiffMode.SIGNED).difference(
            self.ground_truth, self.prediction, DiffMode.ABSOLUTE
        )

        np.testing.assert_allclose(result, [[0.5, 0.5], [1.0, 1.0]])


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

    def test_rejects_non_2d_arrays(self) -> None:
        """Reject 1D (or any non-2D) ground truth/prediction/uncertainty arrays."""
        with pytest.raises(InvalidArrayError, match="Expected 2D"):
            DifferenceCalculator(DiffMode.SIGNED).standardised_difference(
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
            )
