"""Sea ice concentration visualisation module for creating static maps and animations.

- Static map plotting with optional difference visualisation
- Animated video generation (MP4/GIF formats)
- Flexible plotting specifications and colour schemes
- Multiple difference computation strategies
- Date formatting

"""

import logging
from datetime import date, datetime

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL.ImageFile import ImageFile

from icenet_mp.types import ArrayHW, PlotSpec, UncertaintyArrays

from .convert import image_from_figure
from .helpers import (
    _build_title_static,
    _draw_frame,
    _draw_warning_badge,
    _format_title,
    _maybe_add_footer,
    _prepare_difference,
    _prepare_static_plot,
)
from .land_mask import LandMask
from .layout import (
    _add_colourbars,
    _set_axes_limits,
    _set_titles,
    build_layout,
    build_single_panel_figure,
    format_linear_ticks,
    format_symmetric_ticks,
    set_suptitle_with_box,
)
from .plotting_core import (
    colourmap_with_bad,
    compute_standardised_difference,
    create_normalisation,
    style_for_variable,
)

logger = logging.getLogger(__name__)


def plot_static_prediction(  # noqa: PLR0913
    ground_truth: ArrayHW,
    prediction: ArrayHW,
    *,
    date: date | datetime,
    land_mask: LandMask,
    plot_spec: PlotSpec,
    variable_name: str,
    climatology: ArrayHW | None = None,
) -> dict[str, list[ImageFile]]:
    """Create static maps comparing ground truth and prediction data.

    Create static maps and (optional) the difference. The plots use contour mapping with
    customisable colour schemes and include proper axis scaling and colourbars.

    Args:
        ground_truth: 2D array of ground truth sea ice concentration values.
        prediction: 2D array of predicted sea ice concentration values.
        plot_spec: Configuration object specifying titles, colourmaps, value ranges, and
            other visualisation parameters.
        land_mask: Land mask to apply to the data.
        date: Date/datetime for the data being visualised, used in the plot title.
        variable_name: Name of the variable being plotted, used in the plot title and
            output key.
        climatology: Optional 2D array of the calendar-day-mean (climatology) field for
            the calendar day of the date. When given, an additional standalone
            single-panel figure is returned under the key
            ``<date>-<variable_name>-climatology``.

    Returns:
        Dictionary that maps plot names to lists of PIL ImageFile objects. Returns a
        single key `variable_name`-`date` containing a list with one image representing
        the generated plot, plus (when a climatology field is given) the key
        ``<date>-<variable_name>-climatology``.

    Raises:
        InvalidArrayError: If ground_truth and prediction arrays have incompatible shapes.

    """
    (
        height,
        width,
        layout_config,
        warnings,
    ) = _prepare_static_plot(plot_spec, ground_truth, prediction)

    # Initialise the figure and axes with dynamic top spacing if needed
    fig, axs, cbar_axes = build_layout(
        plot_spec=plot_spec,
        height=height,
        width=width,
        layout_config=layout_config,
    )

    # Prepare difference rendering parameters if needed
    difference, diff_colour_scale = _prepare_difference(
        plot_spec, ground_truth, prediction
    )

    # Draw the ground truth and prediction map images
    image_groundtruth, image_prediction, image_difference, _ = _draw_frame(
        axs,
        ground_truth,
        prediction,
        plot_spec,
        land_mask,
        diff_colour_scale=diff_colour_scale,
        precomputed_difference=difference,
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
        cbar_axes=cbar_axes,
    )

    _set_axes_limits(axs, width=width, height=height)
    try:
        title_text = set_suptitle_with_box(
            fig, _build_title_static(variable_name, plot_spec, date)
        )
    except Exception:
        logger.exception("Failed to draw suptitle; continuing without title.")
        title_text = None

    _draw_warning_badge(fig, title_text, warnings)
    _maybe_add_footer(fig, plot_spec)

    try:
        images: dict[str, list[ImageFile]] = {
            f"{date.strftime(r'%Y-%m-%d')}-{variable_name}": [
                image_from_figure(fig, dpi=plot_spec.dpi)
            ]
        }
    finally:
        plt.close(fig)

    if climatology is not None:
        try:
            images.update(
                plot_static_climatology(
                    climatology,
                    date=date,
                    land_mask=land_mask,
                    plot_spec=plot_spec,
                    variable_name=variable_name,
                )
            )
        except (ValueError, OSError) as err:
            logger.warning(
                "Failed to render climatology plot; continuing without it: %s", err
            )

    return images


def plot_static_uncertainty(
    arrays: UncertaintyArrays,
    *,
    date: date | datetime,
    land_mask: LandMask,
    plot_spec: PlotSpec,
    variable_name: str,
) -> dict[str, list[ImageFile]]:
    """Plot signed prediction error in units of observational uncertainty.

    A value of ``z=1`` means the observation exceeds the prediction by one
    reported standard uncertainty. Invalid or non-positive uncertainty values
    are masked.
    """
    z_difference = compute_standardised_difference(
        arrays.ground_truth, arrays.prediction, arrays.uncertainty
    )
    z_difference = land_mask.apply_to(z_difference)

    height, width = z_difference.shape
    fig, ax, cax = build_single_panel_figure(
        height=height,
        width=width,
        colourbar_location=plot_spec.colourbar_location,
    )

    norm, vmin, vmax = create_normalisation(z_difference, centre=0.0)
    image = ax.imshow(
        z_difference,
        cmap=colourmap_with_bad("RdBu_r", bad_color="lightgrey"),
        norm=norm,
        origin="lower",
        interpolation="nearest",
    )

    is_vertical = plot_spec.colourbar_location == "vertical"
    colourbar = fig.colorbar(
        image,
        ax=ax,
        cax=cax,
        orientation=plot_spec.colourbar_location,
    )
    format_symmetric_ticks(
        colourbar,
        vmin=vmin,
        vmax=vmax,
        decimals=1,
        is_vertical=is_vertical,
        centre=0.0,
        use_scientific_notation=False,
    )

    shown = date.date().isoformat() if isinstance(date, datetime) else date.isoformat()
    set_suptitle_with_box(
        fig,
        f"{variable_name} standardised difference (z)   Shown: {shown}",
    )

    try:
        image_file = image_from_figure(fig, dpi=plot_spec.dpi)
    finally:
        plt.close(fig)

    return {f"{shown}-{variable_name}-uncertainty-z": [image_file]}


def plot_static_climatology(
    climatology: ArrayHW,
    *,
    date: date | datetime,
    land_mask: LandMask,
    plot_spec: PlotSpec,
    variable_name: str,
) -> dict[str, list[ImageFile]]:
    """Create a static map of the climatology (calendar-day mean) field.

    Rendered as a standalone single-panel figure, styled like the raw input plots, so it
    can be shown alongside the ground truth/prediction comparison.

    Args:
        climatology: 2D array of the calendar-day-mean (climatology) field.
        plot_spec: Configuration object specifying titles, colourmaps, value ranges, and
            other visualisation parameters.
        land_mask: Land mask to apply to the data.
        date: Date/datetime of the forecast, used in the plot title and output key.
        variable_name: Name of the variable the climatology was computed from, used in
            the plot title and output key.

    Returns:
        Dictionary that maps the plot name ``<date>-<variable_name>-climatology`` to a
        list with one PIL ImageFile for the climatology panel. Returns an empty
        dictionary if the array is not 2D.

    """
    images = plot_static_inputs(
        {variable_name: climatology},
        land_mask=land_mask,
        plot_spec=plot_spec,
        when=date,
    )
    if not images:
        return {}
    image_list = next(iter(images.values()))
    return {f"{date.strftime(r'%Y-%m-%d')}-{variable_name}-climatology": image_list}


def plot_static_inputs(
    variables: dict[str, ArrayHW],
    *,
    land_mask: LandMask,
    plot_spec: PlotSpec,
    when: date | datetime,
) -> dict[str, list[ImageFile]]:
    """Plot one image per input channel as a static map.

    Returns list of (name, list[PIL.ImageFile]).
    """
    results: dict[str, list[ImageFile]] = {}

    for variable_name, variable_values in variables.items():
        if variable_values.ndim != 2:  # noqa: PLR2004
            msg = f"Expected 2D [H,W] for channel '{variable_name}', got {variable_values.shape}"
            logger.warning(msg)
            continue

        # Apply land mask (mask out land to NaN)
        masked_variable_values = land_mask.apply_to(variable_values)

        # Construct the style for this variable
        style = style_for_variable(variable_name, plot_spec.per_variable_styles)

        # Create normalisation using shared function
        norm, vmin, vmax = create_normalisation(
            masked_variable_values,
            vmin=style.vmin,
            vmax=style.vmax,
            centre=style.two_slope_centre,
        )

        # Build figure and axis
        fig, ax, cax = build_single_panel_figure(
            height=masked_variable_values.shape[0],
            width=masked_variable_values.shape[1],
            colourbar_location=plot_spec.colourbar_location,
        )

        # Render
        cmap_name = style.cmap or plot_spec.colourmap
        cmap = colourmap_with_bad(cmap_name, bad_color="lightgrey")
        origin = style.origin or "lower"
        image = ax.imshow(
            masked_variable_values,
            cmap=cmap,
            norm=norm,
            origin=origin,
            interpolation="nearest",
        )

        # Colourbar
        orientation = plot_spec.colourbar_location
        cbar = fig.colorbar(image, ax=ax, cax=cax, orientation=orientation)
        is_vertical = orientation == "vertical"
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

        # Title
        try:
            set_suptitle_with_box(
                fig,
                _format_title(variable_name, plot_spec.hemisphere, when, style.units),
            )
        except (ValueError, AttributeError, RuntimeError) as err:
            logger.debug(
                "Failed to draw static inputs title: %s; continuing without title.", err
            )

        try:
            pil_img = image_from_figure(fig, dpi=plot_spec.dpi)
        finally:
            plt.close(fig)

        # Add image to results dict
        key = f"{when.strftime(r'%Y-%m-%d')}-{variable_name}"
        if key not in results:
            results[key] = []
        results[key].append(pil_img)

    return results
