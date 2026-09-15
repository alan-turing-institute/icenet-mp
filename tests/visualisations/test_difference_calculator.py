"""Tests for icenet_mp/visualisations/difference_calculator.py.

Covers the difference and standardised-difference computations for a
ground-truth/prediction pair.
"""

import numpy as np
import pytest

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.visualisations.difference_calculator import (
    DifferenceCalculator,
)


class TestComputeDifference:
    ground_truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    prediction = np.array([[0.5, 2.5], [2.0, 5.0]])

    def test_signed(self) -> None:
        """Signed difference is ground_truth - prediction."""
        result = DifferenceCalculator().difference(
            self.ground_truth, self.prediction, "signed"
        )

        np.testing.assert_allclose(result, [[0.5, -0.5], [1.0, -1.0]])

    def test_absolute(self) -> None:
        """Absolute difference is |ground_truth - prediction|."""
        result = DifferenceCalculator().difference(
            self.ground_truth, self.prediction, "absolute"
        )

        np.testing.assert_allclose(result, [[0.5, 0.5], [1.0, 1.0]])

    def test_smape(self) -> None:
        """SMAPE difference normalises the absolute error by the mean magnitude."""
        result = DifferenceCalculator().difference(
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

        result = DifferenceCalculator().difference(ground_truth, prediction, "smape")

        assert np.isfinite(result).all()

    def test_invalid_mode_raises(self) -> None:
        """An unrecognised difference mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid difference mode"):
            DifferenceCalculator().difference(
                self.ground_truth,
                self.prediction,
                "bogus",  # type: ignore[arg-type]
            )


class TestComputeStandardisedDifference:
    def test_basic(self) -> None:
        """Compute signed prediction error in units of uncertainty."""
        ground_truth = np.array([[0.5, 0.8], [0.4, 0.2]], dtype=np.float32)
        prediction = np.array([[0.4, 0.6], [0.3, 0.5]], dtype=np.float32)
        uncertainty = np.array([[0.1, 0.2], [0.05, 0.1]], dtype=np.float32)

        result = DifferenceCalculator().standardised_difference(
            ground_truth, prediction, uncertainty
        )

        np.testing.assert_allclose(result, [[1.0, 1.0], [2.0, -3.0]])

    def test_masks_invalid_uncertainty(self) -> None:
        """Mask values where uncertainty is zero, negative or non-finite."""
        ground_truth = np.ones((2, 2), dtype=np.float32)
        prediction = np.zeros((2, 2), dtype=np.float32)
        uncertainty = np.array([[0.5, 0.0], [-1.0, np.nan]], dtype=np.float32)

        result = DifferenceCalculator().standardised_difference(
            ground_truth, prediction, uncertainty
        )

        assert result[0, 0] == pytest.approx(2.0)
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 0])
        assert np.isnan(result[1, 1])

    def test_rejects_shape_mismatch(self) -> None:
        """Reject input arrays with mismatched shapes."""
        with pytest.raises(InvalidArrayError, match="matching shapes"):
            DifferenceCalculator().standardised_difference(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((3, 3), dtype=np.float32),
            )

    def test_rejects_non_2d_arrays(self) -> None:
        """Reject 1D (or any non-2D) ground truth/prediction/uncertainty arrays."""
        with pytest.raises(InvalidArrayError, match="Expected 2D"):
            DifferenceCalculator().standardised_difference(
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
            )
