import numpy as np

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import DiffMode

_SPATIAL_NDIM = 2


class DifferenceCalculator:
    """Computes ground-truth/prediction error fields."""

    def difference(
        self, ground_truth: np.ndarray, prediction: np.ndarray, diff_mode: DiffMode
    ) -> np.ndarray:
        """Compute the difference between the ground truth and prediction.

        Args:
            ground_truth: The ground truth array. [T,H,W]
            prediction: The prediction array. [T,H,W]
            diff_mode: Method to compute the difference.

        Returns:
            Difference array. [T,H,W]

        Raises:
            ValueError: If the difference mode is invalid.

        """
        if diff_mode == "signed":
            return ground_truth - prediction
        if diff_mode == "absolute":
            return np.abs(ground_truth - prediction)
        if diff_mode == "smape":
            denom = np.clip(
                (np.abs(ground_truth) + np.abs(prediction)) / 2.0, 1e-6, None
            )
            return np.abs(prediction - ground_truth) / denom
        msg = f"Invalid difference mode: {diff_mode}"
        raise ValueError(msg)

    def standardised_difference(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        uncertainty: np.ndarray,
    ) -> np.ndarray:
        """Return prediction error in units of observational standard uncertainty.

        The signed convention matches the existing plotting difference convention:
        ``ground_truth - prediction``. Locations with non-finite or non-positive
        uncertainty are returned as NaN because a z value is undefined there.

        Args:
            ground_truth: Two-dimensional observed values.
            prediction: Two-dimensional predicted values.
            uncertainty: Two-dimensional standard uncertainty for the observations.

        Returns:
            Two-dimensional standardised difference array.

        Raises:
            InvalidArrayError: If inputs are not two-dimensional arrays of equal shape.

        """
        arrays = (ground_truth, prediction, uncertainty)
        if any(array.ndim != _SPATIAL_NDIM for array in arrays):
            shapes = tuple(array.shape for array in arrays)
            msg = f"Expected 2D ground truth, prediction and uncertainty arrays, got {shapes}."
            raise InvalidArrayError(msg)
        if not (ground_truth.shape == prediction.shape == uncertainty.shape):
            msg = (
                "Ground truth, prediction and uncertainty must have matching shapes; "
                f"got {ground_truth.shape}, {prediction.shape} and {uncertainty.shape}."
            )
            raise InvalidArrayError(msg)

        result = np.full(ground_truth.shape, np.nan, dtype=float)
        valid = np.isfinite(uncertainty) & (uncertainty > 0)
        np.divide(
            self.difference(ground_truth, prediction, "signed"),
            uncertainty,
            out=result,
            where=valid,
        )
        return result
