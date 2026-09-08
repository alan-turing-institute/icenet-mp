import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import ArrayHW, DiffColourmapSpec, PlotSpec

from .difference_calculator import DifferenceCalculator
from .land_mask import LandMask
from .layout import LayoutConfig, TitleFooterConfig
from .range_checker import RangeChecker
from .variable_styler import VariableStyler


class FrameRenderer:
    """Prepares and draws ground-truth/prediction/difference comparison frames."""

    def prepare_static_plot(
        self,
        plot_spec: PlotSpec,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
    ) -> tuple[int, int, LayoutConfig | None, list[str]]:
        """Validate arrays and compute helpers needed for plotting.

        Returns:
            Tuple of (height, width, layout_config, warnings).

        """
        if ground_truth.shape != prediction.shape:
            msg = f"Prediction ({prediction.shape}) has a different shape to ground truth ({ground_truth.shape})."
            raise InvalidArrayError(msg)
        height, width = ground_truth.shape

        (gt_min, gt_max), (_pred_min, _pred_max) = (
            DifferenceCalculator().compute_display_ranges(
                ground_truth, prediction, plot_spec
            )
        )
        range_check_report = RangeChecker().check(
            ground_truth,
            prediction,
            vmin=gt_min,
            vmax=gt_max,
            outside_warn=getattr(plot_spec, "outside_warn", 0.05),
            severe_outside=getattr(plot_spec, "severe_outside", 0.20),
            include_shared_range_mismatch_check=getattr(
                plot_spec, "include_shared_range_mismatch_check", True
            ),
        )
        warnings = range_check_report.warnings
        layout_config = None
        if warnings:
            layout_config = LayoutConfig(
                title_footer=TitleFooterConfig(title_space=0.10)
            )

        return height, width, layout_config, warnings

    def prepare_difference(
        self,
        plot_spec: PlotSpec,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
    ) -> tuple[np.ndarray | None, DiffColourmapSpec | None]:
        """Compute difference arrays and colour scales if requested."""
        if not plot_spec.include_difference:
            return None, None
        difference = DifferenceCalculator().compute_difference(
            ground_truth, prediction, plot_spec.diff_mode
        )
        diff_colour_scale = DifferenceCalculator().make_diff_colourmap(
            difference, mode=plot_spec.diff_mode
        )
        return difference, diff_colour_scale

    def draw_frame(  # noqa: PLR0913
        self,
        axs: list,
        ground_truth: ArrayHW,
        prediction: ArrayHW,
        plot_spec: PlotSpec,
        land_mask: LandMask,
        *,
        diff_colour_scale: DiffColourmapSpec | None = None,
        precomputed_difference: np.ndarray | None = None,
        levels_override: np.ndarray | None = None,
        display_ranges_override: tuple[tuple[float, float], tuple[float, float]]
        | None = None,
    ) -> tuple:
        """Draw a complete visualisation frame with ground truth, prediction, and optional difference.

        Creates contour plots for ground truth and prediction data, and optionally computes
        and displays their difference. It handles colour normalisation, contour levels, and
        proper cleanup of previous plot elements. Can overlay grey land areas using a land mask.

        Args:
            axs: List of matplotlib axes objects where plots will be drawn. Expected to have
                at least 2 axes (ground truth, prediction) and optionally 3 (with difference).
            ground_truth: 2D array of ground truth values to plot.
            prediction: 2D array of predicted values to plot.
            plot_spec: Plotting specification containing colourmap, value ranges, and other
                display parameters.
            land_mask: LandMask instance for applying land area overlays in grey.
            diff_colour_scale: Optional DiffColourmapSpec containing normalisation and colour
                mapping parameters for difference plots.
            precomputed_difference: Optional pre-computed difference array. If None and
                difference is included, the difference will be computed on-demand.
            levels_override: Optional custom contour levels. If None, levels are derived
                from plot_spec.
            display_ranges_override: Optional custom display ranges for stable animation.
                If None, ranges are computed from the data.

        Returns:
            Tuple containing (image_groundtruth, image_prediction, image_difference, diff_colour_scale).
            The image objects are matplotlib contour collections, and image_difference may be None
            if include_difference is False. diff_colour_scale is returned for reuse in animations.

        """
        for ax in axs:
            self._clear_plot(ax)

        # Compute difference if required
        difference = (
            precomputed_difference
            if precomputed_difference is not None
            else DifferenceCalculator().compute_difference(
                ground_truth, prediction, plot_spec.diff_mode
            )
        )

        # Apply land mask to data if provided
        ground_truth = land_mask.apply_to(ground_truth)
        prediction = land_mask.apply_to(prediction)
        difference = land_mask.apply_to(difference)

        # Compute display ranges - use override if provided for stable animation
        display_ranges = (
            DifferenceCalculator().compute_display_ranges(
                ground_truth, prediction, plot_spec
            )
            if display_ranges_override is None
            else display_ranges_override
        )

        # Draw ground truth and prediction panels
        image_groundtruth, image_prediction = self._draw_main_panels(
            axs,
            ground_truth,
            prediction,
            plot_spec,
            display_ranges,
            levels_override,
        )

        image_difference = None
        if plot_spec.include_difference:
            # Expect colour scale to be provided by caller; no fallback here
            if diff_colour_scale is None:
                error_msg = (
                    "diff_colour_scale must be provided when including difference"
                )
                raise InvalidArrayError(error_msg)

            # Create colourmap with bad color handling for NaN values
            diff_cmap = VariableStyler().colourmap_with_bad(
                diff_colour_scale.cmap, bad_color="lightgrey"
            )

            if diff_colour_scale.norm is not None:
                # Signed differences with TwoSlopeNorm - use explicit levels to ensure consistency
                diff_vmin = diff_colour_scale.norm.vmin or 0.0
                diff_vmax = diff_colour_scale.norm.vmax or 1.0
                diff_levels = self._safe_linspace(
                    diff_vmin, diff_vmax, plot_spec.n_contour_levels
                )

                image_difference = axs[2].contourf(
                    difference,
                    levels=diff_levels,
                    cmap=diff_cmap,
                    vmin=diff_vmin,
                    vmax=diff_vmax,
                    origin="lower",
                )
            else:
                # Non-negative differences with vmin/vmax
                vmin = diff_colour_scale.vmin or 0.0
                vmax = diff_colour_scale.vmax or 1.0
                diff_levels = self._safe_linspace(
                    vmin,
                    vmax,
                    plot_spec.n_contour_levels,
                )

                image_difference = axs[2].contourf(
                    difference,
                    levels=diff_levels,
                    cmap=diff_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                )

        # Optional: visually mark NaNs as semi-transparent grey overlays
        def _overlay_nans(ax: Axes, arr: np.ndarray, land_color: str = "white") -> None:
            """Overlay NaNs as semi-transparent grey overlays.

            Not used currently, add the following before the return statement to enable:
            >>> _overlay_nans(axs[0], ground_truth)
            >>> _overlay_nans(axs[1], prediction)
            >>> if plot_spec.include_difference:
            >>>     _overlay_nans(axs[2], difference)
            Mask should then be visible in the plot.

            """
            if np.isnan(arr).any():
                # Create overlay for NaN areas (land mask)
                nan_mask = np.isnan(arr).astype(float)
                # Create a custom colourmap: 0=transparent, 1=land color

                # Land colour options: typically 'white' or 'black' (or any valid Matplotlib color)
                colors = ["white", land_color]  # 0=white (transparent), 1=land color
                cmap = ListedColormap(colors)
                ax.imshow(
                    nan_mask,
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    alpha=1.0,  # Fully opaque white land areas
                    interpolation="nearest",
                )

        # Optional: visually mark NaNs as semi-transparent grey overlays here
        _overlay_nans(axs[0], ground_truth, land_color="white")
        _overlay_nans(axs[1], prediction, land_color="white")
        if plot_spec.include_difference and difference is not None:
            # Use black for the difference panel so zero (white) remains distinguishable from land
            _overlay_nans(axs[2], difference, land_color="black")

        return image_groundtruth, image_prediction, image_difference, diff_colour_scale

    def _draw_main_panels(
        self,
        axs: list,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        plot_spec: PlotSpec,
        display_ranges: tuple[tuple[float, float], tuple[float, float]],
        levels_override: np.ndarray | None = None,
    ) -> tuple:
        """Draw ground truth and prediction panels.

        Args:
            axs: List of matplotlib axes objects.
            ground_truth: Ground truth data array.
            prediction: Prediction data array.
            plot_spec: Plotting specification.
            display_ranges: Tuple of (vmin, vmax) for ground truth and prediction.
            levels_override: Optional custom contour levels.

        Returns:
            Tuple of (image_groundtruth, image_prediction).

        """
        # Create colourmap with bad color handling for NaN values
        cmap = VariableStyler().colourmap_with_bad(
            plot_spec.colourmap, bad_color="lightgrey"
        )

        # Expand display_ranges tuple for clarity
        (groundtruth_vmin, groundtruth_vmax), (prediction_vmin, prediction_vmax) = (
            display_ranges
        )

        # Set the contour colour levels
        if levels_override is not None:
            groundtruth_levels = prediction_levels = levels_override
        elif plot_spec.colourbar_strategy == "separate":
            # For separate strategy, use explicit levels to prevent breathing
            groundtruth_levels = self._safe_linspace(
                groundtruth_vmin, groundtruth_vmax, plot_spec.n_contour_levels
            )
            prediction_levels = self._safe_linspace(
                prediction_vmin, prediction_vmax, plot_spec.n_contour_levels
            )
        else:
            # For shared strategy, use same levels for both panels
            groundtruth_levels = prediction_levels = self._safe_linspace(
                min(groundtruth_vmin, prediction_vmin),
                max(groundtruth_vmax, prediction_vmax),
                plot_spec.n_contour_levels,
            )

        image_groundtruth = axs[0].contourf(
            ground_truth,
            levels=groundtruth_levels,
            cmap=cmap,
            vmin=groundtruth_vmin,
            vmax=groundtruth_vmax,
            origin="lower",
        )
        image_prediction = axs[1].contourf(
            prediction,
            levels=prediction_levels,
            cmap=cmap,
            vmin=prediction_vmin,
            vmax=prediction_vmax,
            origin="lower",
        )

        return image_groundtruth, image_prediction

    def _clear_plot(self, ax: Axes) -> None:
        """Remove titles, labels, and contour collections from an axes to prevent overlaps.

        Prevents overlapping plots in an animation loop.

        Args:
            ax: Matplotlib axes object to clear.

        """
        for coll in ax.collections[:]:
            coll.remove()
        ax.set_title("")

    def _safe_linspace(self, vmin: float, vmax: float, n: int) -> np.ndarray:
        """Return an increasing linspace even if vmin==vmax or the inputs are swapped."""
        # ensure float
        vmin = float(vmin)
        vmax = float(vmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            # fallback to a stable interval
            vmin, vmax = 0.0, 1.0
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if vmax == vmin:
            # create a tiny range so contourf accepts the levels
            eps = max(1e-6, abs(vmin) * 1e-6)
            vmax = vmin + eps
        return np.linspace(vmin, vmax, n)
