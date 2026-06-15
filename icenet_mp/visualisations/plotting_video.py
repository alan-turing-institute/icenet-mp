"""Raw input variable visualisation for a single timestep or animation over time.

Creates one static map per input channel with optional per-variable style overrides,
or animations showing temporal evolution of individual variables.

This module is intentionally lightweight and single-panel oriented while reusing the
layout and formatting helpers from the sea-ice plotting stack to keep appearances aligned.
"""

import io
import logging
from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import ArrayTHW, PlotSpec
from icenet_mp.visualisations.layout import (
    build_single_panel_figure,
    format_linear_ticks,
    format_symmetric_ticks,
    set_suptitle_with_box,
)
from icenet_mp.visualisations.plotting_core import (
    colourmap_with_bad,
    compute_difference,
    compute_display_ranges,
    create_normalisation,
    levels_from_spec,
    make_diff_colourmap,
    prepare_difference_stream,
    safe_nanmax,
    safe_nanmin,
    style_for_variable,
)

from .convert import video_from_animation
from .helpers import (
    _build_footer_video,
    _build_title_video,
    _draw_frame,
    _format_title,
)
from .land_mask import LandMask
from .layout import (
    LayoutConfig,
    TitleFooterConfig,
    _add_colourbars,
    _set_axes_limits,
    _set_titles,
    build_layout,
    set_footer_with_box,
)

if TYPE_CHECKING:
    from matplotlib.text import Text

logger = logging.getLogger(__name__)


def plot_video_prediction(
    ground_truth: ArrayTHW,
    prediction: ArrayTHW,
    *,
    dates: Sequence[date | datetime],
    land_mask: LandMask,
    plot_spec: PlotSpec,
    variable_name: str = "sea-ice-concentration",
) -> dict[str, io.BytesIO]:
    """Generate animations showing the temporal evolution of sea ice concentration.

    Compares ground truth data with model predictions. Supports multiple video formats
    and difference computation strategies for optimal performance with large datasets.

    Args:
        ground_truth: 3D array with shape (time, height, width) containing
            ground truth sea ice concentration values over time.
        prediction: 3D array with shape (time, height, width) containing
            predicted sea ice concentration values. Must have the same shape as
            ground_truth_stream.
        dates: List of date/datetime objects corresponding to each timestep.
            Must have the same length as the time dimension of the data arrays.
        land_mask: Land mask to apply to the data.
        plot_spec: Configuration object specifying titles, colourmaps, value ranges, and
            other visualisation parameters.
        variable_name: Name of the variable being plotted, used in the plot title and
            output key.

    Returns:
        Dictionary mapping video names to BytesIO objects containing the encoded video
        data. Currently this returns a single key `variable_name`-`date` containing the
        video data suitable for direct wandb.Video logging.

    Raises:
        InvalidArrayError: If arrays have incompatible shapes or if the number of
            dates doesn't match the number of timesteps.
        VideoRenderError: If video encoding fails.

    """
    # Check the shapes of the arrays
    n_timesteps, height, width = ground_truth.shape
    if ground_truth.shape != prediction.shape:
        msg = f"Prediction ({prediction.shape}) has a different shape to ground truth ({ground_truth.shape})."
        raise InvalidArrayError(msg)
    if len(dates) != ground_truth.shape[0]:
        error_msg = (
            f"Number of dates ({len(dates)}) != number of timesteps ({n_timesteps})"
        )
        raise InvalidArrayError(error_msg)

    # Apply land mask to data streams
    masked_ground_truth = land_mask.apply_to(ground_truth)
    masked_prediction = land_mask.apply_to(prediction)

    # Initialise the figure and axes with a larger footer space for videos
    layout_config = LayoutConfig(title_footer=TitleFooterConfig(footer_space=0.11))
    fig, axs, cbar_axes = build_layout(
        plot_spec=plot_spec,
        height=height,
        width=width,
        layout_config=layout_config,
    )
    levels = levels_from_spec(plot_spec)

    # Stable ranges for the whole animation
    display_ranges = compute_display_ranges(
        masked_ground_truth, masked_prediction, plot_spec
    )

    # Prepare the difference array according to the strategy; also ensure a colour scale exists
    difference_stream, diff_colour_scale = prepare_difference_stream(
        include_difference=plot_spec.include_difference,
        diff_mode=plot_spec.diff_mode,
        strategy=plot_spec.diff_strategy,
        ground_truth_stream=masked_ground_truth,
        prediction_stream=masked_prediction,
    )
    if plot_spec.include_difference and diff_colour_scale is None:
        # Per-frame strategy: infer a stable colour scale from the first frame to avoid breathing
        first_diff = compute_difference(
            masked_ground_truth[0],
            masked_prediction[0],
            plot_spec.diff_mode,
        )
        diff_colour_scale = make_diff_colourmap(first_diff, mode=plot_spec.diff_mode)
    precomputed_diff_0 = (
        difference_stream[0]
        if (plot_spec.include_difference and difference_stream is not None)
        else None
    )

    # Initial frame
    image_groundtruth, image_prediction, image_difference, _ = _draw_frame(
        axs,
        masked_ground_truth[0],
        masked_prediction[0],
        plot_spec,
        land_mask,
        diff_colour_scale=diff_colour_scale,
        precomputed_difference=precomputed_diff_0,
        levels_override=levels,
        display_ranges_override=display_ranges,
    )
    # Restore axis titles after drawing (they were cleared in _draw_frame)
    _set_titles(axs, plot_spec)
    # Colourbars and title
    _add_colourbars(
        axs,
        image_groundtruth=image_groundtruth,
        image_prediction=image_prediction,
        image_difference=image_difference,
        plot_spec=plot_spec,
        diff_colour_scale=diff_colour_scale,
        display_ranges=display_ranges,
        cbar_axes=cbar_axes,
    )
    _set_axes_limits(axs, width=width, height=height)
    try:
        title_text = set_suptitle_with_box(
            fig, _build_title_video(variable_name, plot_spec, dates, 0)
        )
    except Exception:
        logger.exception("Failed to draw suptitle; continuing without title.")
        title_text = None

    # Footer metadata at the bottom (static across frames)
    if getattr(plot_spec, "include_footer_metadata", True):
        try:
            footer_text = _build_footer_video(plot_spec, dates)
            if footer_text:
                set_footer_with_box(fig, footer_text)
        except Exception:
            logger.exception("Failed to draw footer; continuing without footer.")

    # Animation function
    def animate(tt: int) -> tuple[()]:
        precomputed_diff_tt = (
            difference_stream[tt]
            if (plot_spec.include_difference and difference_stream is not None)
            else None
        )
        _draw_frame(
            axs,
            masked_ground_truth[tt],
            masked_prediction[tt],
            plot_spec,
            land_mask,
            diff_colour_scale=diff_colour_scale,
            precomputed_difference=precomputed_diff_tt,
            levels_override=levels,
            display_ranges_override=display_ranges,
        )
        # Restore axis titles after drawing (they were cleared in _draw_frame)
        _set_titles(axs, plot_spec)

        if title_text is not None:
            title_text.set_text(_build_title_video(variable_name, plot_spec, dates, tt))
        return ()

    try:
        # Create the animation object
        animation_object = animation.FuncAnimation(
            fig,
            animate,
            frames=n_timesteps,
            interval=1000 // plot_spec.video_fps,
            blit=False,
            repeat=True,
        )
        # Write to BytesIO buffer
        video_buffer = video_from_animation(
            animation_object,
            fps=plot_spec.video_fps,
            video_format=plot_spec.video_format,
        )
        # Return the video buffer
        return {f"{variable_name}-{dates[0].strftime(r'%Y-%m-%d')}": video_buffer}
    finally:
        # Clean up by closing figure
        plt.close(fig)


def plot_video_single_input(
    variable_name: str,
    variable_values: ArrayTHW,
    *,
    dates: Sequence[date | datetime],
    land_mask: LandMask,
    plot_spec: PlotSpec,
) -> io.BytesIO:
    """Create animation showing temporal evolution of a single variable.

    This function creates a video animation of a single input variable over time,
    with consistent colour scaling across all frames to avoid visual "breathing".

    Args:
        dates: Sequence of dates corresponding to each timestep (length must match T).
        land_mask: Optional 2D boolean array marking land areas [H, W].
        plot_spec: Base plotting specification (colourmap, hemisphere, etc.).
        variable_name: Name of the variable (e.g., "era5:2t", "osisaf-south:ice_conc").
        variable_values: 3D array of data over time [T, H, W].

    Returns:
        BytesIO buffer containing the encoded video data.

    Raises:
        InvalidArrayError: If data_stream is not 3D or dates length mismatches.
        VideoRenderError: If video encoding fails.

    """
    # Validate input
    if variable_values.ndim != 3:  # noqa: PLR2004
        msg = f"Expected 3D data [T,H,W], got shape {variable_values.shape}"
        raise InvalidArrayError(msg)

    n_timesteps, height, width = variable_values.shape
    if len(dates) != n_timesteps:
        msg = f"Number of dates ({len(dates)}) != number of timesteps ({n_timesteps})"
        raise InvalidArrayError(msg)

    # Apply land mask
    masked_variable_values = land_mask.apply_to(variable_values)

    # Get styling for this variable
    style = style_for_variable(variable_name, plot_spec.per_variable_styles)

    # Create stable normalisation across all frames
    # Use global min/max to prevent colour scale "breathing"
    data_min = safe_nanmin(masked_variable_values)
    data_max = safe_nanmax(masked_variable_values)

    # Override with style limits if provided
    effective_vmin = style.vmin if style.vmin is not None else data_min
    effective_vmax = style.vmax if style.vmax is not None else data_max

    # Create normalisation using first frame to establish type
    norm, vmin, vmax = create_normalisation(
        masked_variable_values[0],
        vmin=effective_vmin,
        vmax=effective_vmax,
        centre=style.two_slope_centre,
    )

    # Build figure with single panel + colourbar
    fig, ax, cax = build_single_panel_figure(
        height=height,
        width=width,
        colourbar_location=plot_spec.colourbar_location,
    )

    # Render initial frame
    cmap_name = style.cmap or plot_spec.colourmap
    cmap = colourmap_with_bad(cmap_name, bad_color="lightgrey")
    origin = style.origin or "lower"
    image = ax.imshow(
        masked_variable_values[0],
        cmap=cmap,
        norm=norm,
        origin=origin,
        interpolation="nearest",
    )

    # Create colourbar (stable across all frames)
    orientation = plot_spec.colourbar_location
    cbar = fig.colorbar(image, ax=ax, cax=cax, orientation=orientation)
    is_vertical = orientation == "vertical"

    # Format colourbar ticks based on normalisation type
    decimals = style.decimals if style.decimals is not None else 2
    use_scientific = (
        style.use_scientific_notation
        if style.use_scientific_notation is not None
        else False
    )
    if isinstance(norm, TwoSlopeNorm):
        format_symmetric_ticks(
            cbar,
            vmin=vmin,
            vmax=vmax,
            decimals=decimals,
            is_vertical=is_vertical,
            centre=norm.vcenter,
            use_scientific_notation=use_scientific,
        )
    else:
        format_linear_ticks(
            cbar,
            vmin=vmin,
            vmax=vmax,
            decimals=decimals,
            is_vertical=is_vertical,
            use_scientific_notation=use_scientific,
        )

    # Create title (will be updated each frame)
    title_text: Text | None = None
    try:
        title_text = set_suptitle_with_box(
            fig,
            _format_title(variable_name, plot_spec.hemisphere, dates[0], style.units),
        )
    except (ValueError, AttributeError, RuntimeError) as err:
        logger.debug("Failed to draw title: %s; continuing without title.", err)

    # Animation update function
    def animate(tt: int) -> tuple[()]:
        """Update function for each frame of the animation."""
        # Update image data
        image.set_data(masked_variable_values[tt])
        # Update title with current date
        if title_text is not None:
            title_text.set_text(
                _format_title(
                    variable_name, plot_spec.hemisphere, dates[tt], style.units
                )
            )
        return ()

    try:
        # Create animation object
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=n_timesteps,
            interval=1000 // plot_spec.video_fps,
            blit=False,
            repeat=True,
        )
        # Return a BytesIO video buffer
        return video_from_animation(
            anim, fps=plot_spec.video_fps, video_format=plot_spec.video_format
        )
    finally:
        # Clean up by closing figure
        plt.close(fig)


def plot_video_inputs(
    variables: dict[str, ArrayTHW],
    *,
    dates: Sequence[date | datetime],
    land_mask: LandMask,
    plot_spec: PlotSpec,
) -> dict[str, io.BytesIO]:
    """Create animations for multiple input variables over time.

    This is a convenience wrapper around `plot_video_single_input()` that
    processes multiple variables in a batch, applying consistent styling and
    land masking to all variables.

    Args:
        dates: Sequence of dates for each timestep (length must match T).
        plot_spec: Plotting specification.
        land_mask: Land mask to apply to all variables.
        variables: Dictionary of variable name to THW 3D array of values.

    Returns:
        Dictionary mapping `variable_name`-`date` to video_buffer.

    Raises:
        InvalidArrayError: If arrays/names count mismatch or invalid shapes.

    """
    results: dict[str, io.BytesIO] = {}

    # for data_stream, var_name in zip(channel_arrays_stream, channel_names, strict=True):
    for variable_name, variable_values in variables.items():
        logger.debug("Creating animation for variable: %s", variable_name)

        try:
            # Create animation for this variable
            video_buffer = plot_video_single_input(
                variable_name,
                variable_values,
                dates=dates,
                land_mask=land_mask,
                plot_spec=plot_spec,
            )
            results[f"{variable_name}-{dates[0].strftime(r'%Y-%m-%d')}"] = video_buffer

        except (InvalidArrayError, ValueError, MemoryError, OSError):
            logger.exception(
                "Failed to create animation for variable %s, skipping", variable_name
            )
            continue

    return results
