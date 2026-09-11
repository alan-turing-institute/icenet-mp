"""Domain-specific panel assembly on top of the minimal render_panels core.

`PanelRenderer` takes raw ground-truth/prediction/input arrays and applies masking
and (where relevant) difference or standardised-difference panels, rendering the
result via `render_panels_static` or `render_panels_video`. `render.py` itself stays
domain-agnostic (arrays and styles in, image/video out); this module is where land
masks, difference modes and uncertainty become panels.

Used by both `Plotter` (production logging) and `dataset_plotting.py` (CLI dataset
preview plots), each of which builds one `PanelRenderer` per land_mask/plot_spec
pairing rather than threading those two through every call.
"""

from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from PIL.ImageFile import ImageFile

from icenet_mp.types import ArrayHW, ArrayTHW, PlotSpec

from .difference_calculator import DifferenceCalculator
from .land_mask import LandMask
from .plot_annotator import PlotAnnotator
from .render import render_panels_static, render_panels_video
from .variable_styler import VariableStyler

if TYPE_CHECKING:
    from matplotlib.colors import Colormap, Normalize


class PanelRenderer:
    """Renders styled, land-masked panels for one land_mask/plot_spec pairing.

    Owns the plot_spec-driven collaborators (`VariableStyler`, `PlotAnnotator`,
    `DifferenceCalculator`) so callers don't need to construct or coordinate
    them directly.
    """

    def __init__(self, land_mask: LandMask, plot_spec: PlotSpec) -> None:
        """Build a renderer bound to one land mask and plot spec."""
        self.land_mask = land_mask
        self.plot_spec = plot_spec
        self._variable_styler = VariableStyler()
        self._annotator = PlotAnnotator()
        self._difference_calculator = DifferenceCalculator()

    def static_singlet(
        self,
        values: ArrayHW,
        *,
        when: datetime,
        variable_name: str,
    ) -> ImageFile:
        """Render a single static input panel via render_panels_static."""
        plot_spec = self.plot_spec
        masked_values = self.land_mask.apply_to(values)
        style = self._variable_styler.style_for_variable(
            variable_name, plot_spec.per_variable_styles
        )
        title = self._annotator.format_title(
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

    def video_singlet(
        self,
        values: ArrayTHW,
        *,
        dates: list[datetime],
        variable_name: str,
    ) -> BytesIO:
        """Render a single video input panel via render_panels_video."""
        plot_spec = self.plot_spec
        masked_values = self.land_mask.apply_to(values)
        style = self._variable_styler.style_for_variable(
            variable_name, plot_spec.per_variable_styles
        )
        title = self._annotator.format_title(
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

    def _difference_panel(
        self, masked_ground_truth: np.ndarray, masked_prediction: np.ndarray
    ) -> tuple[np.ndarray, str, str, float | None, float | None]:
        """Compute the difference panel array, title, cmap and colour bounds.

        Shared by `static_triplet` and `video_triplet`, which otherwise each
        rebuilt this identically for their (single) difference panel.
        """
        plot_spec = self.plot_spec
        difference = self.land_mask.apply_to(
            self._difference_calculator.compute_difference(
                masked_ground_truth, masked_prediction, plot_spec.diff_mode
            )
        )
        diff_colour_scale = self._difference_calculator.make_diff_colourmap(
            difference, mode=plot_spec.diff_mode
        )
        if diff_colour_scale.norm is not None:
            diff_vmin = diff_colour_scale.norm.vmin
            diff_vmax = diff_colour_scale.norm.vmax
        else:
            diff_vmin = diff_colour_scale.vmin
            diff_vmax = diff_colour_scale.vmax
        title = f"{plot_spec.title_difference} ({plot_spec.diff_mode})"
        return difference, title, diff_colour_scale.cmap, diff_vmin, diff_vmax

    def static_triplet(
        self,
        ground_truth: ArrayHW,
        prediction: ArrayHW,
        *,
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
        plot_spec = self.plot_spec
        masked_ground_truth = self.land_mask.apply_to(ground_truth)
        masked_prediction = self.land_mask.apply_to(prediction)

        arrays = [masked_ground_truth, masked_prediction]
        titles = [plot_spec.title_groundtruth, plot_spec.title_prediction]
        cmaps: list[str | Colormap] = [plot_spec.colourmap, plot_spec.colourmap]
        norms: list[Normalize | None] = [None, None]
        vmins: list[float | None] = [plot_spec.vmin, plot_spec.vmin]
        vmaxs: list[float | None] = [plot_spec.vmax, plot_spec.vmax]

        # If we have uncertainty data then use z-score as the third panel
        if uncertainty is not None:
            z_difference = self.land_mask.apply_to(
                self._difference_calculator.compute_standardised_difference(
                    ground_truth, prediction, uncertainty
                )
            )
            z_norm, _, _ = self._variable_styler.create_normalisation(
                z_difference, centre=0.0
            )

            arrays.append(z_difference)
            titles.append("Standardised Difference (z)")
            cmaps.append(
                self._variable_styler.colourmap_with_bad(
                    "RdBu_r", bad_color="lightgrey"
                )
            )
            norms.append(z_norm)
            vmins.append(None)
            vmaxs.append(None)

        # Otherwise, use the difference panel if requested
        elif plot_spec.include_difference:
            difference, title, cmap, diff_vmin, diff_vmax = self._difference_panel(
                masked_ground_truth, masked_prediction
            )
            arrays.append(difference)
            titles.append(title)
            cmaps.append(cmap)
            norms.append(None)
            vmins.append(diff_vmin)
            vmaxs.append(diff_vmax)

        suptitle = self._annotator.title_for_static(variable_name, plot_spec, when)
        footer_text = self._annotator.footer_for_static(plot_spec)
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

    def video_triplet(
        self,
        ground_truth: ArrayTHW,
        prediction: ArrayTHW,
        *,
        dates: list[datetime],
        variable_name: str,
    ) -> BytesIO:
        """Render the ground-truth/prediction(/difference) triptych video via render_panels_video."""
        plot_spec = self.plot_spec
        masked_ground_truth = self.land_mask.apply_to(ground_truth)
        masked_prediction = self.land_mask.apply_to(prediction)

        arrays = [masked_ground_truth, masked_prediction]
        titles = [plot_spec.title_groundtruth, plot_spec.title_prediction]
        cmaps: list[str] = [plot_spec.colourmap, plot_spec.colourmap]
        vmins: list[float | None] = [plot_spec.vmin, plot_spec.vmin]
        vmaxs: list[float | None] = [plot_spec.vmax, plot_spec.vmax]

        if plot_spec.include_difference:
            difference, title, cmap, diff_vmin, diff_vmax = self._difference_panel(
                masked_ground_truth, masked_prediction
            )
            arrays.append(difference)
            titles.append(title)
            cmaps.append(cmap)
            vmins.append(diff_vmin)
            vmaxs.append(diff_vmax)

        title_line = self._annotator.title_for_video(variable_name, plot_spec, dates, 0)
        footer_text = self._annotator.footer_for_video(plot_spec, dates)

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
