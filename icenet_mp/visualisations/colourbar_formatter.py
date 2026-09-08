"""Colourbar creation and tick formatting for map comparison layouts."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.contour import QuadContourSet
from matplotlib.ticker import FuncFormatter

from icenet_mp.types import DiffColourmapSpec, PlotSpec

from .layout import LayoutConfig


class ColourbarFormatter:
    """Creates and tick-formats colourbars for map comparison layouts."""

    def add_colourbars(  # noqa: PLR0913, PLR0912
        self,
        axs: list[Axes],
        *,
        image_groundtruth: QuadContourSet,
        image_prediction: QuadContourSet | None = None,
        image_difference: QuadContourSet | None = None,
        plot_spec: PlotSpec,
        diff_colour_scale: DiffColourmapSpec | None = None,
        display_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
        cbar_axes: dict[str, Axes | None] | None = None,
    ) -> None:
        """Create and position colourbars.

        This function creates two types of colourbars:
        1. Shared colourbar for ground truth and prediction (same data scale)
        2. Separate colourbar for difference data (different scale, often symmetric)

        The function works with the layout from LayoutBuilder.build_layout, using dedicated
        colourbar axes when available, or falling back to automatic matplotlib placement.

        Colourbar Design:
            - Ground truth/prediction: Uses data range from plot_spec
            - Difference: Often symmetric around zero with specialised tick formatting
            - Orientation: Matches plot_spec.colourbar_location ('vertical'/'horizontal')

        Args:
            axs: List of main plot axes in order [ground_truth, prediction, difference*]
                 (*difference only present if include_difference=True)
            image_groundtruth: QuadContourSet from ground truth plot - provides colour mapping
                              and normalisation for shared ground truth/prediction colourbar
            image_prediction: Optional QuadContourSet from prediction plot.
            image_difference: Optional QuadContourSet from difference plot - provides
                             specialised colour mapping (often symmetric) for difference colourbar
            plot_spec: PlotSpec containing colourbar orientation, value ranges, and formatting
            diff_colour_scale: Optional difference colour scale specification.
            display_ranges: Optional tuple of (groundtruth_range, prediction_range) for colourbar limits.
            cbar_axes: Optional dict with pre-allocated colourbar axes from layout builders:
                      {"groundtruth": Axes|None, "prediction": Axes|None, "difference": Axes|None}
                      None values mean "no dedicated axis" and will fall back to automatic placement.

        Layout:
            - cbar_axes["prediction"]: Used for shared ground truth/prediction colourbar
            - cbar_axes["difference"]: Used for difference panel colourbar

        """
        orientation = plot_spec.colourbar_location
        is_vertical = orientation == "vertical"
        separate_colourbars = plot_spec.colourbar_strategy == "separate"

        # Expand display_ranges tuple for clarity
        groundtruth_vmin = display_ranges[0][0] if display_ranges else None
        groundtruth_vmax = display_ranges[0][1] if display_ranges else None
        prediction_vmin = display_ranges[1][0] if display_ranges else None
        prediction_vmax = display_ranges[1][1] if display_ranges else None

        if separate_colourbars:
            # Create individual colourbars for each panel

            # Ground truth colourbar
            if cbar_axes and cbar_axes.get("groundtruth"):
                colourbar_groundtruth = plt.colorbar(
                    image_groundtruth,
                    cax=cbar_axes["groundtruth"],
                    orientation=orientation,
                )
                self.format_linear_ticks(
                    colourbar_groundtruth,
                    vmin=groundtruth_vmin,
                    vmax=groundtruth_vmax,
                    decimals=1,
                    is_vertical=is_vertical,
                )

            # Prediction colourbar
            if image_prediction and cbar_axes and cbar_axes.get("prediction"):
                colourbar_prediction = plt.colorbar(
                    image_prediction,
                    cax=cbar_axes["prediction"],
                    orientation=orientation,
                )
                self.format_linear_ticks(
                    colourbar_prediction,
                    vmin=prediction_vmin,
                    vmax=prediction_vmax,
                    decimals=1,
                    is_vertical=is_vertical,
                )
        else:
            # Create shared colourbar for ground truth and prediction panels
            if cbar_axes is not None and cbar_axes.get("prediction") is not None:
                # Use dedicated colourbar axis (preferred - better layout control)
                colourbar_truth = plt.colorbar(
                    image_groundtruth,
                    cax=cbar_axes["prediction"],
                    orientation=orientation,
                )
            else:
                # Fallback: automatic positioning across both panels
                colourbar_truth = plt.colorbar(
                    image_groundtruth, ax=[axs[0], axs[1]], orientation=orientation
                )

            # Tick formatting
            self.format_linear_ticks(
                colourbar_truth,
                vmin=groundtruth_vmin,
                vmax=groundtruth_vmax,
                decimals=1,
                is_vertical=is_vertical,
            )

        # Create separate colourbar for difference panel (if present)
        # Difference data often has different scale and symmetric range around zero
        if plot_spec.include_difference and image_difference is not None:
            # Build a fixed ScalarMappable so the colourbar never depends on per-frame QuadContourSet
            sm = None
            if diff_colour_scale is not None:
                if diff_colour_scale.norm is not None:
                    sm = plt.cm.ScalarMappable(
                        norm=diff_colour_scale.norm, cmap=diff_colour_scale.cmap
                    )
                else:
                    sm = plt.cm.ScalarMappable(
                        norm=Normalize(
                            vmin=diff_colour_scale.vmin, vmax=diff_colour_scale.vmax
                        ),
                        cmap=diff_colour_scale.cmap,
                    )
                sm.set_array([])

            if cbar_axes is not None and cbar_axes.get("difference") is not None:
                colourbar_diff = plt.colorbar(
                    sm if sm is not None else image_difference,
                    cax=cbar_axes["difference"],
                    orientation=orientation,
                )
            else:
                colourbar_diff = plt.colorbar(
                    sm if sm is not None else image_difference,
                    ax=axs[2],
                    orientation=orientation,
                )

            # Tick formatting: symmetric for TwoSlopeNorm, otherwise linear
            if isinstance(image_difference.norm, TwoSlopeNorm):
                vmin = float(
                    image_difference.norm.vmin
                    or LayoutConfig().formatting.default_vmin_diff_fallback
                )
                vmax = float(
                    image_difference.norm.vmax
                    or LayoutConfig().formatting.default_vmax_diff_fallback
                )
                self.format_symmetric_ticks(
                    colourbar_diff,
                    vmin=vmin,
                    vmax=vmax,
                    decimals=2,
                    is_vertical=is_vertical,
                    centre=image_difference.norm.vcenter,
                )
            else:
                self.format_linear_ticks(
                    colourbar_diff, decimals=2, is_vertical=is_vertical
                )

    def format_linear_ticks(
        self,
        colourbar: Colorbar,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        decimals: int = 1,
        is_vertical: bool,
        use_scientific_notation: bool = False,
    ) -> None:
        """Format a linear colourbar with 5 ticks.

        If vmin/vmax are not provided, derive them from the colourbar's mappable.

        Args:
            colourbar: Colorbar to format.
            vmin: Minimum value.
            vmax: Maximum value.
            decimals: Number of decimal places for tick labels.
            is_vertical: Whether the colorbar is vertical.
            use_scientific_notation: Whether to format tick labels in scientific notation.

        """
        axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis

        if vmin is None or vmax is None:
            mvmin, mvmax = self.get_cbar_limits_from_mappable(colourbar)
            vmin = mvmin if vmin is None else vmin
            vmax = mvmax if vmax is None else vmax

        ticks = np.linspace(
            float(vmin), float(vmax), LayoutConfig().formatting.num_ticks_linear
        )
        colourbar.set_ticks([float(t) for t in ticks])

        if use_scientific_notation:
            axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}e}"))
        else:
            axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}f}"))
        if not is_vertical:
            colourbar.ax.xaxis.set_tick_params(pad=1)
        self._apply_monospace_to_cbar_text(colourbar)

    def format_symmetric_ticks(  # noqa: PLR0913
        self,
        colourbar: Colorbar,
        *,
        vmin: float,
        vmax: float,
        decimals: int = 2,
        is_vertical: bool,
        centre: float | None = None,
        use_scientific_notation: bool = False,
    ) -> None:
        """Format symmetric diverging ticks with a centred midpoint.

        Places five ticks: [vmin, midpoint to centre, centre, centre to midpoint, vmax].

        Args:
            colourbar: Colorbar to format.
            vmin: Minimum value.
            vmax: Maximum value.
            decimals: Number of decimal places for tick labels.
            is_vertical: Whether the colorbar is vertical.
            centre: centre value for diverging colourmap (default: 0.0).
            use_scientific_notation: Whether to format tick labels in scientific notation.

        """
        axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis
        centre_val = centre if centre is not None else 0.0
        midpoint_factor = LayoutConfig().formatting.midpoint_factor
        ticks = [
            vmin,
            midpoint_factor * (vmin + centre_val),
            centre_val,
            midpoint_factor * (centre_val + vmax),
            vmax,
        ]
        colourbar.set_ticks([float(t) for t in ticks])

        if use_scientific_notation:
            axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}e}"))
        else:
            axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}f}"))
        if not is_vertical:
            colourbar.ax.xaxis.set_tick_params(pad=1)
        self._apply_monospace_to_cbar_text(colourbar)

    def get_cbar_limits_from_mappable(self, cbar: Colorbar) -> tuple[float, float]:
        """Return (vmin, vmax) for a colourbar's mappable with robust fallbacks."""
        vmin = vmax = None
        try:  # Works for many matplotlib mappables
            vmin, vmax = cbar.mappable.get_clim()  # type: ignore[attr-defined]
        except AttributeError:
            norm = getattr(cbar.mappable, "norm", None)
            vmin = getattr(norm, "vmin", None)
            vmax = getattr(norm, "vmax", None)
        if vmin is None or vmax is None:
            vmin, vmax = (
                LayoutConfig().formatting.default_vmin_fallback,
                LayoutConfig().formatting.default_vmax_fallback,
            )
        return float(vmin), float(vmax)

    def _apply_monospace_to_cbar_text(self, colourbar: Colorbar) -> None:
        """Set tick labels and axis labels on a colourbar to monospace family."""
        ax = colourbar.ax
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            label.set_fontfamily("monospace")
        # Ensure axis labels also use monospace if present
        ax.xaxis.label.set_fontfamily("monospace")
        ax.yaxis.label.set_fontfamily("monospace")
