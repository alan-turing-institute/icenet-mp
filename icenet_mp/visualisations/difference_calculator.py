import numpy as np
from matplotlib.colors import TwoSlopeNorm

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import DiffColourmapSpec, DiffMode
from icenet_mp.utils import safe_nanmax, safe_nanmin

_SPATIAL_NDIM = 2


class DifferenceCalculator:
    """Computes differences and display/colour ranges for GT/prediction pairs."""

    def compute_difference(
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

    def compute_standardised_difference(
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
            self.compute_difference(ground_truth, prediction, "signed"),
            uncertainty,
            out=result,
            where=valid,
        )
        return result

    def make_diff_colourmap(
        self,
        sample: np.ndarray | float,
        *,
        mode: DiffMode,
    ) -> DiffColourmapSpec:
        """Construct colour mapping settings for a difference panel.

        Behaviour depends on the difference mode:

        - "signed": symmetric diverging scale centred on 0,
          useful for showing positive vs negative bias.
        - "absolute" / "smape": sequential scale from 0 to max,
          useful for showing error magnitude.

        Args:
            sample: Either a full array of differences (for precompute mode)
                    or a scalar maximum difference (for two-pass mode).
            mode: Difference mode ("signed", "absolute", or "smape").

        Returns:
            DiffRenderParams: Normalisation, colour limits, and colourmap.

        """
        if mode == "signed":
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

            return DiffColourmapSpec(
                norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
                vmin=None,
                vmax=None,
                cmap="RdBu_r",
            )

        if mode in ("absolute", "smape"):
            # Positive-only scale
            if isinstance(sample, (float, int)):
                vmax = max(1e-6, float(sample))
            else:
                vmax = max(1e-6, safe_nanmax(sample, default=0.0))

            return DiffColourmapSpec(
                norm=None,
                vmin=0.0,
                vmax=vmax,
                cmap="magma",
            )

        msg = f"Unknown difference mode: {mode}"
        raise ValueError(msg)
