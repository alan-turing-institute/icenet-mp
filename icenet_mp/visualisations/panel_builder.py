"""Domain-specific panel assembly on top of the minimal render_panels core.

Each function here takes raw ground-truth/prediction/input arrays plus a
`LandMask` and `PlotSpec`, applies masking and (where relevant) difference or
standardised-difference panels, and renders the result via `render_panels_static`
or `render_panels_video`. `render.py` itself stays domain-agnostic (arrays and
styles in, image/video out); this module is where land masks, difference modes
and uncertainty become panels.

Used by both `Plotter` (production logging) and `icenet_mp.synthetic.debug_video`
(standalone debug videos), so the functions here take `land_mask`/`plot_spec`
explicitly rather than reading them off `self`.
"""

from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING

from PIL.ImageFile import ImageFile

from icenet_mp.types import ArrayHW, ArrayTHW, PlotSpec

from .difference_calculator import DifferenceCalculator
from .land_mask import LandMask
from .plot_annotator import PlotAnnotator
from .render import render_panels_static, render_panels_video
from .variable_styler import VariableStyler

if TYPE_CHECKING:
    from matplotlib.colors import Colormap, Normalize


def render_static_singlet(
    values: ArrayHW,
    *,
    land_mask: LandMask,
    plot_spec: PlotSpec,
    when: datetime,
    variable_name: str,
) -> ImageFile:
    """Render a single static input panel via render_panels_static."""
    masked_values = land_mask.apply_to(values)
    style = VariableStyler().style_for_variable(
        variable_name, plot_spec.per_variable_styles
    )
    title = PlotAnnotator().format_title(
        variable_name, plot_spec.hemisphere, when, style.units
    )
    return render_panels_static(
        [masked_values],
        cmap=style.cmap or plot_spec.colourmap,
        dpi=plot_spec.dpi,
        figure_title=title,
        vmax=style.vmax,
        vmin=style.vmin,
    )


def render_video_singlet(
    values: ArrayTHW,
    *,
    dates: list[datetime],
    land_mask: LandMask,
    plot_spec: PlotSpec,
    variable_name: str,
) -> BytesIO:
    """Render a single video input panel via render_panels_video."""
    masked_values = land_mask.apply_to(values)
    style = VariableStyler().style_for_variable(
        variable_name, plot_spec.per_variable_styles
    )
    title = PlotAnnotator().format_title(
        variable_name, plot_spec.hemisphere, dates[0], style.units
    )
    return render_panels_video(
        [masked_values],
        cmap=style.cmap or plot_spec.colourmap,
        dpi=plot_spec.dpi,
        figure_title=title,
        fps=plot_spec.video_fps,
        vmax=style.vmax,
        vmin=style.vmin,
        video_format=plot_spec.video_format,
    )


def render_static_triplet(  # noqa: PLR0913
    ground_truth: ArrayHW,
    prediction: ArrayHW,
    *,
    land_mask: LandMask,
    plot_spec: PlotSpec,
    when: datetime,
    variable_name: str,
    uncertainty: ArrayHW | None = None,
) -> ImageFile:
    """Render the ground-truth/prediction(/difference)(/uncertainty) panels via render_panels.

    When `uncertainty` is provided, an extra panel shows the standardised
    difference `z = (ground_truth - prediction) / uncertainty`. A value of
    `z=1` means the observation exceeds the prediction by one reported
    standard uncertainty.
    """
    masked_ground_truth = land_mask.apply_to(ground_truth)
    masked_prediction = land_mask.apply_to(prediction)

    arrays = [masked_ground_truth, masked_prediction]
    titles = [plot_spec.title_groundtruth, plot_spec.title_prediction]
    cmaps: list[str | Colormap] = [plot_spec.colourmap, plot_spec.colourmap]
    norms: list[Normalize | None] = [None, None]
    vmins: list[float | None] = [plot_spec.vmin, plot_spec.vmin]
    vmaxs: list[float | None] = [plot_spec.vmax, plot_spec.vmax]

    # If we have uncertainty data then use z-score as the third panel
    if uncertainty is not None:
        variable_styler = VariableStyler()
        z_difference = land_mask.apply_to(
            DifferenceCalculator().compute_standardised_difference(
                ground_truth, prediction, uncertainty
            )
        )
        z_norm, _, _ = variable_styler.create_normalisation(z_difference, centre=0.0)

        arrays.append(z_difference)
        titles.append("Standardised Difference (z)")
        cmaps.append(
            variable_styler.colourmap_with_bad("RdBu_r", bad_color="lightgrey")
        )
        norms.append(z_norm)
        vmins.append(None)
        vmaxs.append(None)

    # Otherwise, use the difference panel if requested
    elif plot_spec.include_difference:
        difference_calculator = DifferenceCalculator()
        difference = land_mask.apply_to(
            difference_calculator.compute_difference(
                masked_ground_truth, masked_prediction, plot_spec.diff_mode
            )
        )
        diff_colour_scale = difference_calculator.make_diff_colourmap(
            difference, mode=plot_spec.diff_mode
        )
        if diff_colour_scale.norm is not None:
            diff_vmin = diff_colour_scale.norm.vmin
            diff_vmax = diff_colour_scale.norm.vmax
        else:
            diff_vmin = diff_colour_scale.vmin
            diff_vmax = diff_colour_scale.vmax

        arrays.append(difference)
        titles.append(f"{plot_spec.title_difference} ({plot_spec.diff_mode})")
        cmaps.append(diff_colour_scale.cmap)
        norms.append(None)
        vmins.append(diff_vmin)
        vmaxs.append(diff_vmax)

    annotator = PlotAnnotator()
    suptitle = annotator.title_for_static(variable_name, plot_spec, when)
    footer_text = annotator.footer_for_static(plot_spec)
    return render_panels_static(
        arrays,
        cmap=cmaps,
        dpi=plot_spec.dpi,
        figure_title=suptitle,
        footer_text=footer_text or None,
        group_axes=(0, 1)
        if plot_spec.include_difference or uncertainty is not None
        else None,
        norm=norms,
        panel_titles=titles,
        vmax=vmaxs,
        vmin=vmins,
    )


def render_video_triplet(
    ground_truth: ArrayTHW,
    prediction: ArrayTHW,
    *,
    dates: list[datetime],
    land_mask: LandMask,
    plot_spec: PlotSpec,
    variable_name: str,
) -> BytesIO:
    """Render the ground-truth/prediction(/difference) triptych video via render_panels_video."""
    masked_ground_truth = land_mask.apply_to(ground_truth)
    masked_prediction = land_mask.apply_to(prediction)

    arrays = [masked_ground_truth, masked_prediction]
    titles = [plot_spec.title_groundtruth, plot_spec.title_prediction]
    cmaps: list[str] = [plot_spec.colourmap, plot_spec.colourmap]
    vmins: list[float | None] = [plot_spec.vmin, plot_spec.vmin]
    vmaxs: list[float | None] = [plot_spec.vmax, plot_spec.vmax]

    if plot_spec.include_difference:
        difference_calculator = DifferenceCalculator()
        difference = land_mask.apply_to(
            difference_calculator.compute_difference(
                masked_ground_truth, masked_prediction, plot_spec.diff_mode
            )
        )
        diff_colour_scale = difference_calculator.make_diff_colourmap(
            difference, mode=plot_spec.diff_mode
        )
        if diff_colour_scale.norm is not None:
            diff_vmin = diff_colour_scale.norm.vmin
            diff_vmax = diff_colour_scale.norm.vmax
        else:
            diff_vmin = diff_colour_scale.vmin
            diff_vmax = diff_colour_scale.vmax

        arrays.append(difference)
        titles.append(f"{plot_spec.title_difference} ({plot_spec.diff_mode})")
        cmaps.append(diff_colour_scale.cmap)
        vmins.append(diff_vmin)
        vmaxs.append(diff_vmax)

    annotator = PlotAnnotator()
    title_line = annotator.title_for_video(variable_name, plot_spec, dates, 0)
    footer_text = annotator.footer_for_video(plot_spec, dates)

    return render_panels_video(
        arrays,
        cmap=cmaps,
        dpi=plot_spec.dpi,
        figure_title=title_line,
        footer_text=footer_text or None,
        fps=plot_spec.video_fps,
        group_axes=(0, 1) if plot_spec.include_difference else None,
        panel_titles=titles,
        vmax=vmaxs,
        vmin=vmins,
        video_format=plot_spec.video_format,
    )
