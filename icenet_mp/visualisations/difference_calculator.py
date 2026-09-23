import numpy as np
from matplotlib.colors import TwoSlopeNorm

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import ColourStyle, DiffMode
from icenet_mp.utils import safe_nanmax, safe_nanmin

_VALID_NDIMS = (2, 3)


class DifferenceCalculator:
    """Compute a difference field from ground-truth/prediction base fields."""

    def __init__(self, diff_mode: DiffMode) -> None:
        """Initialise a DifferenceCalculator with a difference mode.

        Args:
            diff_mode: Which method to use to compute differences.

        """
        self.diff_mode = diff_mode

    def colour_style(
        self,
        sample: np.ndarray | float,
    ) -> ColourStyle:
        """Construct a ColourStyle for visualising differences.

        This function generates a ColourStyle object that contains the appropriate
        normalisation, colour limits, and colourmap for visualising differences between
        datasets. The behaviour of the colour mapping depends on the difference mode
        bound at construction.

        Args:
            sample: A full array of differences

        Returns:
            ColourStyle: Normalisation, colour limits, and colourmap for the difference panel.

        """
        if self.diff_mode == DiffMode.SIGNED:
            # Force symmetric limits around zero so 0 is the literal midpoint
            if isinstance(sample, (float, int)):
                max_abs = max(1.0, float(abs(sample)))
                vmin, vmax = -max_abs, max_abs
            else:
                # Find the min and max values of the sample array using safe helpers
                vmin_data = safe_nanmin(sample, default=-1.0)
                vmax_data = safe_nanmax(sample, default=1.0)
                # Find the maximum absolute value of the sample array
                max_abs = max(1.0, abs(vmin_data), abs(vmax_data))
                vmin, vmax = -max_abs, max_abs

            return ColourStyle(
                norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
                vmin=None,
                vmax=None,
                cmap="RdBu_r",
            )

        if self.diff_mode in (DiffMode.ABSOLUTE, DiffMode.SMAPE):
            # Positive-only scale
            if isinstance(sample, (float, int)):
                vmax = max(1e-6, float(sample))
            else:
                vmax = max(1e-6, safe_nanmax(sample, default=0.0))

            return ColourStyle(
                norm=None,
                vmin=0.0,
                vmax=vmax,
                cmap="magma",
            )

        msg = f"Unknown difference mode: {self.diff_mode}"
        raise ValueError(msg)

    def difference(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
    ) -> np.ndarray:
        """Compute the difference between the ground truth and prediction.

        Args:
            ground_truth: The ground truth array. [T,H,W]
            prediction: The prediction array. [T,H,W]

        Returns:
            Difference array. [T,H,W]

        Raises:
            ValueError: If the difference mode is invalid.

        """
        if ground_truth.shape != prediction.shape:
            msg = (
                "Ground truth and prediction must have matching shapes; "
                f"got {ground_truth.shape} and {prediction.shape}."
            )
            raise InvalidArrayError(msg)
        if self.diff_mode == DiffMode.SIGNED:
            return ground_truth - prediction
        if self.diff_mode == DiffMode.ABSOLUTE:
            return np.abs(ground_truth - prediction)
        if self.diff_mode == DiffMode.SMAPE:
            denom = np.clip(
                (np.abs(ground_truth) + np.abs(prediction)) / 2.0, 1e-6, None
            )
            return np.abs(prediction - ground_truth) / denom
        msg = f"Unknown difference mode: {self.diff_mode}"
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
            ground_truth: Observed values, 2D `[H, W]` (static) or 3D `[T, H, W]` (video).
            prediction: Predicted values, matching `ground_truth`'s shape.
            uncertainty: Standard uncertainty for the observations, matching
                `ground_truth`'s shape.

        Returns:
            Standardised difference array, matching `ground_truth`'s shape.

        Raises:
            InvalidArrayError: If inputs are not 2D or 3D arrays of equal shape.

        """
        arrays = (ground_truth, prediction, uncertainty)
        if any(array.ndim not in _VALID_NDIMS for array in arrays):
            shapes = tuple(array.shape for array in arrays)
            msg = (
                "Expected 2D [H, W] or 3D [T, H, W] ground truth, prediction and "
                f"uncertainty arrays, got {shapes}."
            )
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
            self.difference(ground_truth, prediction),
            uncertainty,
            out=result,
            where=valid,
        )
        return result
