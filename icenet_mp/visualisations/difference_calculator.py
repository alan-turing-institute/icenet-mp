import numpy as np
from matplotlib.colors import TwoSlopeNorm

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import DiffColourmapSpec, DiffMode, DiffStrategy, PlotSpec

from .variable_styler import VariableStyler

_SPATIAL_NDIM = 2


class DifferenceCalculator:
    """Computes differences and display/colour ranges for GT/prediction pairs."""

    def __init__(self) -> None:
        """Initialize a DifferenceCalculator."""
        self._variable_styler = VariableStyler()

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
                vmin_data = self._variable_styler.safe_nanmin(sample, default=-1.0)
                vmax_data = self._variable_styler.safe_nanmax(sample, default=1.0)
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
                vmax = max(1e-6, self._variable_styler.safe_nanmax(sample, default=0.0))

            return DiffColourmapSpec(
                norm=None,
                vmin=0.0,
                vmax=vmax,
                cmap="magma",
            )

        msg = f"Unknown difference mode: {mode}"
        raise ValueError(msg)

    def prepare_difference_stream(
        self,
        *,
        include_difference: bool,
        diff_mode: DiffMode,
        strategy: DiffStrategy,
        ground_truth_stream: np.ndarray,
        prediction_stream: np.ndarray,
    ) -> tuple[np.ndarray | None, DiffColourmapSpec | None]:
        """General, reusable planner for animations (maps or time-series).

        This function implements three different strategies for handling
        difference between ground truth and prediction in animations, each with
        different memory and computational trade-offs. The choice of strategy affects
        both memory usage and animation performance.

        Args:
            include_difference: Whether difference visualisation is requested.
            diff_mode: Type of difference computation (signed/absolute/smape).
            strategy: Strategy for difference computation:
                - "precompute": Calculate all differences upfront
                - "two-pass": Scan data to determine colour scale, then compute per-frame
                - "per-frame": Compute differences on-demand
            ground_truth_stream: 3D array of ground truth data over time.
            prediction_stream: 3D array of prediction data over time.

        Returns:
            Tuple of (difference_stream, colour_scale):
            - difference_stream: None unless strategy == 'precompute'
            - colour_scale: DiffColourmapSpec describing colourmap/norm/range

        """
        if not include_difference:
            return None, None

        n_timesteps = ground_truth_stream.shape[0]
        if strategy == "precompute":
            difference_stream = self.compute_difference(
                ground_truth_stream, prediction_stream, diff_mode
            )
            colour_scale = self.make_diff_colourmap(difference_stream, mode=diff_mode)
            return difference_stream, colour_scale

        if strategy == "two-pass":
            differences = [
                self.compute_difference(
                    ground_truth_stream[tt], prediction_stream[tt], diff_mode
                )
                for tt in range(n_timesteps)
            ]
            max_ = max(
                float(
                    np.nanmax(
                        np.abs(difference) if diff_mode == "signed" else difference
                    )
                    or 0.0
                )
                for difference in differences
            )
            colour_scale = self.make_diff_colourmap(max_, mode=diff_mode)
            return None, colour_scale

        if strategy == "per-frame":
            # no precomputation; each frame will call compute_difference and
            # may choose to infer its own params if desired (less consistent look)
            return None, None

        msg = f"Unknown DiffStrategy: {strategy}"
        raise ValueError(msg)

    def compute_display_ranges(
        self, ground_truth: np.ndarray, prediction: np.ndarray, plot_spec: PlotSpec
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Compute vmin/vmax for ground truth and prediction based on strategy.

        Args:
            ground_truth: The ground truth array. [H,W]
            prediction: The prediction array. [H,W]
            plot_spec: The plotting specification.

        Returns:
            The display ranges. (vmin, vmax)

        Raises:
            InvalidArrayError: If the arrays are not 2D or have different shapes.

        """
        # Get data ranges for ground_truth and prediction
        groundtruth_min = float(np.nanmin(ground_truth))
        groundtruth_max = float(np.nanmax(ground_truth))
        prediction_min = float(np.nanmin(prediction))
        prediction_max = float(np.nanmax(prediction))

        # Both panels use the same vmin/vmax (ground truth used as reference)
        if plot_spec.colourbar_strategy == "shared":
            shared_range = (groundtruth_min, groundtruth_max)
            return shared_range, shared_range

        # Each panel uses its own data range ("separate", the only other Literal value)
        return (groundtruth_min, groundtruth_max), (prediction_min, prediction_max)
