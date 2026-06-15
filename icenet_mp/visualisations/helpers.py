import logging
from collections.abc import Sequence
from datetime import date, datetime

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.text import Text

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import ArrayHW, DiffColourmapSpec, PlotSpec

from .land_mask import LandMask
from .layout import (
    LayoutConfig,
    TitleFooterConfig,
    draw_badge_with_box,
    set_footer_with_box,
)
from .plotting_core import (
    colourmap_with_bad,
    compute_difference,
    compute_display_ranges,
    levels_from_spec,
    make_diff_colourmap,
)
from .range_check import compute_range_check_report

#: Default plotting specification for sea ice concentration visualisation
DEFAULT_SIC_SPEC = PlotSpec(
    variable="sea_ice_concentration",
    title_groundtruth="Ground Truth",
    title_prediction="Prediction",
    title_difference="Difference",
    colourmap="viridis",
    diff_mode="signed",
    vmin=0.0,
    vmax=1.0,
    colourbar_location="horizontal",
    colourbar_strategy="shared",
    outside_warn=0.05,
    severe_outside=0.20,
    include_shared_range_mismatch_check=True,
)

logger = logging.getLogger(__name__)


def _safe_linspace(vmin: float, vmax: float, n: int) -> np.ndarray:
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


def _prepare_static_plot(
    plot_spec: PlotSpec,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[int, int, LayoutConfig | None, list[str], np.ndarray]:
    """Validate arrays and compute helpers needed for plotting.

    Returns:
        Tuple of (height, width, layout_config, warnings, contour levels).

    """
    if ground_truth.shape != prediction.shape:
        msg = f"Prediction ({prediction.shape}) has a different shape to ground truth ({ground_truth.shape})."
        raise InvalidArrayError(msg)
    height, width = ground_truth.shape

    (gt_min, gt_max), (_pred_min, _pred_max) = compute_display_ranges(
        ground_truth, prediction, plot_spec
    )
    range_check_report = compute_range_check_report(
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
        layout_config = LayoutConfig(title_footer=TitleFooterConfig(title_space=0.10))

    levels = levels_from_spec(plot_spec)
    return height, width, layout_config, warnings, levels


def _prepare_difference(
    plot_spec: PlotSpec,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray | None, DiffColourmapSpec | None]:
    """Compute difference arrays and colour scales if requested."""
    if not plot_spec.include_difference:
        return None, None
    difference = compute_difference(ground_truth, prediction, plot_spec.diff_mode)
    diff_colour_scale = make_diff_colourmap(difference, mode=plot_spec.diff_mode)
    return difference, diff_colour_scale


def _draw_warning_badge(
    fig: Figure,
    title_text: Text | None,
    warnings: Sequence[str],
) -> None:
    """Render a warning badge close to the title."""
    if not warnings:
        return
    badge = "Warnings: " + ", ".join(warnings)
    if title_text is not None:
        _, title_y = title_text.get_position()
        n_lines = title_text.get_text().count("\n") + 1
        warning_y = max(title_y - (0.05 + 0.02 * (n_lines - 1)), 0.0)
    else:
        warning_y = 0.90
    draw_badge_with_box(fig, 0.5, warning_y, badge)


def _maybe_add_footer(fig: Figure, plot_spec: PlotSpec) -> None:
    """Attach footer metadata when enabled."""
    if not getattr(plot_spec, "include_footer_metadata", True):
        return
    try:
        footer_text = _build_footer_static(plot_spec)
        if footer_text:
            set_footer_with_box(fig, footer_text)
    except Exception:
        logger.exception("Failed to draw footer; continuing without footer.")


def _format_title(
    variable: str, hemisphere: str | None, when: date | datetime, units: str | None
) -> str:
    """Format a title string for a raw input variable plot.

    Args:
        variable: Variable name.
        hemisphere: Hemisphere ("north" or "south"), if applicable.
        when: Date or datetime of the data.
        units: Display units for the variable.

    Returns:
        Formatted title string.

    """
    hemi = f" ({hemisphere.capitalize()})" if hemisphere else ""
    units_s = f" [{units}]" if units else ""
    shown = when.date().isoformat() if isinstance(when, datetime) else when.isoformat()
    return f"{variable}{units_s}{hemi}   Shown: {shown}"


def _draw_main_panels(
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
    cmap = colourmap_with_bad(plot_spec.colourmap, bad_color="lightgrey")

    # Expand display_ranges tuple for clarity
    (groundtruth_vmin, groundtruth_vmax), (prediction_vmin, prediction_vmax) = (
        display_ranges
    )

    # Set the contour colour levels
    if levels_override is not None:
        groundtruth_levels = prediction_levels = levels_override
    elif plot_spec.colourbar_strategy == "separate":
        # For separate strategy, use explicit levels to prevent breathing
        groundtruth_levels = _safe_linspace(
            groundtruth_vmin, groundtruth_vmax, plot_spec.n_contour_levels
        )
        prediction_levels = _safe_linspace(
            prediction_vmin, prediction_vmax, plot_spec.n_contour_levels
        )
    else:
        # For shared strategy, use same levels for both panels
        groundtruth_levels = prediction_levels = _safe_linspace(
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


def _draw_frame(  # noqa: PLR0913
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
        _clear_plot(ax)

    # Compute difference if required
    difference = (
        precomputed_difference
        if precomputed_difference is not None
        else compute_difference(ground_truth, prediction, plot_spec.diff_mode)
    )

    # Apply land mask to data if provided
    ground_truth = land_mask.apply_to(ground_truth)
    prediction = land_mask.apply_to(prediction)
    difference = land_mask.apply_to(difference)

    # Compute display ranges - use override if provided for stable animation
    display_ranges = (
        compute_display_ranges(ground_truth, prediction, plot_spec)
        if display_ranges_override is None
        else display_ranges_override
    )

    # Draw ground truth and prediction panels
    image_groundtruth, image_prediction = _draw_main_panels(
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
            error_msg = "diff_colour_scale must be provided when including difference"
            raise InvalidArrayError(error_msg)

        # Create colourmap with bad color handling for NaN values
        diff_cmap = colourmap_with_bad(diff_colour_scale.cmap, bad_color="lightgrey")

        if diff_colour_scale.norm is not None:
            # Signed differences with TwoSlopeNorm - use explicit levels to ensure consistency
            diff_vmin = diff_colour_scale.norm.vmin or 0.0
            diff_vmax = diff_colour_scale.norm.vmax or 1.0
            diff_levels = _safe_linspace(
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
            diff_levels = _safe_linspace(
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


def _clear_plot(ax: Axes) -> None:
    """Remove titles, labels, and contour collections from an axes to prevent overlaps.

    Prevents overlapping plots in an animation loop.

    Args:
        ax: Matplotlib axes object to clear.

    """
    for coll in ax.collections[:]:
        coll.remove()
    ax.set_title("")


# --- Title helpers ---
def _formatted_variable_name(variable: str) -> str:
    """Return a human-friendly variable name for titles.

    Example: "sea_ice_concentration" -> "Sea ice concentration".
    """
    pretty = variable.replace("_", " ").strip()
    return pretty.title() if pretty else ""


def _format_date_for_title(dt: date | datetime) -> str:
    """Format a date/datetime to ISO date string (YYYY-MM-DD) for plot titles.

    Args:
        dt: Date or datetime object to format.

    Returns:
        ISO format date string (YYYY-MM-DD). Time components are stripped
        from datetime objects.

    Example:
        >>> from datetime import date, datetime
        >>> _format_date_for_title(date(2023, 12, 25))
        '2023-12-25'
        >>> _format_date_for_title(datetime(2023, 12, 25, 14, 30))
        '2023-12-25'

    """
    if isinstance(dt, datetime):
        return dt.date().isoformat()
    return dt.isoformat()


def _build_title_static(
    variable_name: str, plot_spec: PlotSpec, when: date | datetime
) -> str:
    """Compose a simple suptitle for static plots.

    Lines:
      1) "<Variable> (<Hemisphere>)  Shown: YYYY-MM-DD"
         (Footer contains any metadata such as model/epoch/training data if present)
    """
    metric = _formatted_variable_name(variable_name)
    hemi = f" ({plot_spec.hemisphere.capitalize()})" if plot_spec.hemisphere else ""
    return f"{metric}{hemi} Prediction   Shown: {_format_date_for_title(when)}"


def _build_title_video(
    variable_name: str,
    plot_spec: PlotSpec,
    dates: Sequence[date | datetime],
    current_index: int,
) -> str:
    """Compose a simple suptitle for video plots (date changes per frame).

    Lines:
      1) "<Variable> (<Hemisphere>)  Frame: YYYY-MM-DD"
      2) Footer: "Animating from <start> to <end>"
      3) Footer: "Model: <model>  Epoch: <num>  Training Dates: <start> — <end> (<cadence>) <num> pts" (optional)
      4) Footer: "Training Data: <source> (<vars>) <source> (<vars>)" (optional)
    """
    metric = _formatted_variable_name(variable_name)
    hemi = f" ({plot_spec.hemisphere.capitalize()})" if plot_spec.hemisphere else ""
    if dates:
        return f"{metric}{hemi} Prediction   Frame: {_format_date_for_title(dates[current_index])}"
    return f"{metric}{hemi} Prediction"


def _build_footer_static(plot_spec: PlotSpec) -> str:
    """Build footer text for static plots using metadata that used to be in title."""
    lines: list[str] = []
    if plot_spec.metadata_subtitle:
        lines.append(plot_spec.metadata_subtitle)
    return "\n".join(lines)


def _build_footer_video(plot_spec: PlotSpec, dates: Sequence[date | datetime]) -> str:
    """Build footer text for video plots: animation range and metadata."""
    lines: list[str] = []
    if dates:
        start_s = _format_date_for_title(dates[0])
        end_s = _format_date_for_title(dates[-1])
        lines.append(f"Animating from {start_s} to {end_s}")
    if plot_spec.metadata_subtitle:
        lines.append(plot_spec.metadata_subtitle)
    return "\n".join(lines)
