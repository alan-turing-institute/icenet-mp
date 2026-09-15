"""Domain-specific panel assembly on top of the minimal Renderer core.

`PanelRenderer` takes raw ground-truth/prediction/input arrays and applies masking
and (where relevant) difference or standardised-difference panels, rendering the
result via `Renderer.panels_static` or `Renderer.panels_video`.

Used by both `MediaPublisher` (logging during runs) and `DatasetMediaWriter` (CLI
dataset preview plots), each of which builds one `PanelRenderer` per land_mask/plot_spec
pairing.
"""

from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL.ImageFile import ImageFile

from icenet_mp.types import ArrayHW, ArrayTHW, Metadata, PlotSpec

from .colour_scale import ColourScale
from .difference_calculator import DifferenceCalculator
from .land_mask import LandMask
from .plot_annotator import PlotAnnotator
from .renderer import Renderer
from .variable_style_resolver import VariableStyleResolver

if TYPE_CHECKING:
    from matplotlib.colors import Colormap, Normalize


class PanelRenderer:
    """Renders styled, land-masked panels for one land_mask/plot_spec pairing."""

    def __init__(
        self, land_mask: LandMask, metadata: Metadata, plot_spec: PlotSpec
    ) -> None:
        """Build a renderer bound to one land mask and plot spec."""
        self.land_mask = land_mask
        self.plot_spec = plot_spec
        self._style_resolver = VariableStyleResolver(
            plot_spec.per_variable_styles, plot_spec.colourmap
        )
        self._colour_scale = ColourScale(plot_spec.diff_mode)
        self._annotator = PlotAnnotator(metadata, plot_spec)
        self._difference_calculator = DifferenceCalculator(plot_spec.diff_mode)
        self._renderer = Renderer()

    @property
    def video_format(self) -> Literal["mp4", "gif"]:
        return self.plot_spec.video_format

    def static_singlet(
        self,
        values: ArrayHW,
        *,
        when: datetime,
        variable_name: str,
    ) -> ImageFile:
        """Render a single panel ImageFile via Renderer.panels_static()."""
        plot_spec = self.plot_spec
        masked_values = self.land_mask.apply_to(values)
        style = self._style_resolver.style_for_variable(variable_name)
        title = self._annotator.format_title(variable_name, when, style.units)
        return self._renderer.panels_static(
            [masked_values],
            cmap=style.cmap,
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
        """Render a single panel video BytesIO via Renderer.panels_video()."""
        masked_values = self.land_mask.apply_to(values)
        style = self._style_resolver.style_for_variable(variable_name)
        title = self._annotator.format_title(variable_name, dates[0], style.units)
        return self._renderer.panels_video(
            [masked_values],
            cmap=style.cmap,
            dpi=self.plot_spec.dpi,
            figure_title=title,
            fps=self.plot_spec.video_fps,
            vmax=style.vmax,
            vmin=style.vmin,
            video_format=self.video_format,
        )

    def _difference_panel(
        self, masked_ground_truth: np.ndarray, masked_prediction: np.ndarray
    ) -> tuple[np.ndarray, str, str, float | None, float | None]:
        """Compute the difference panel array, title, cmap and colour bounds.

        Shared by `static_triplet` and `video_triplet`, which otherwise each
        rebuilt this identically for their (single) difference panel.
        """
        difference = self.land_mask.apply_to(
            self._difference_calculator.difference(
                masked_ground_truth, masked_prediction
            )
        )
        diff_colour_scale = self._colour_scale.diff_colourmap(difference)
        diff_vmin, diff_vmax = self._colour_scale.bounds(diff_colour_scale)
        title = f"{self.plot_spec.title_difference} ({self.plot_spec.diff_mode})"
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
        """Render a three panel ImageFile via Renderer.panels().

        Args:
            ground_truth: 2D array of the ground truth field.
            prediction: 2D array of the predicted field.
            when: Datetime of the plotted timestep.
            variable_name: Name of the variable being plotted, used for styling and
                title generation.
            uncertainty: Optional 2D array of the reported standard uncertainty of the
                prediction field. When given, an additional panel shows the standardised
                difference `z = (ground_truth - prediction) / uncertainty`. A value of
                `z=1` means the observation exceeds the prediction by one reported standard
                uncertainty.

        """
        masked_ground_truth = self.land_mask.apply_to(ground_truth)
        masked_prediction = self.land_mask.apply_to(prediction)

        arrays = [masked_ground_truth, masked_prediction]
        titles = [self.plot_spec.title_groundtruth, self.plot_spec.title_prediction]
        cmaps: list[str | Colormap] = [
            self.plot_spec.colourmap,
            self.plot_spec.colourmap,
        ]
        norms: list[Normalize | None] = [None, None]
        vmins: list[float | None] = [self.plot_spec.vmin, self.plot_spec.vmin]
        vmaxs: list[float | None] = [self.plot_spec.vmax, self.plot_spec.vmax]

        # If we have uncertainty data then use z-score as the third panel
        if uncertainty is not None:
            z_difference = self.land_mask.apply_to(
                self._difference_calculator.standardised_difference(
                    ground_truth, prediction, uncertainty
                )
            )
            z_norm = self._colour_scale.normalisation(z_difference, centre=0.0)

            arrays.append(z_difference)
            titles.append("Standardised Difference (z)")
            cmaps.append(self._colour_scale.colourmap("RdBu_r", bad_color="lightgrey"))
            norms.append(z_norm)
            vmins.append(None)
            vmaxs.append(None)

        # Otherwise, use the difference panel if requested
        elif self.plot_spec.include_difference:
            difference, title, cmap, diff_vmin, diff_vmax = self._difference_panel(
                masked_ground_truth, masked_prediction
            )
            arrays.append(difference)
            titles.append(title)
            cmaps.append(cmap)
            norms.append(None)
            vmins.append(diff_vmin)
            vmaxs.append(diff_vmax)

        contour_arrays: list[np.ndarray | None] | None = None
        if self.plot_spec.include_ice_edge:
            contour_arrays = [masked_ground_truth, masked_prediction]
            contour_arrays += [None] * (len(arrays) - len(contour_arrays))

        suptitle = self._annotator.title_for_static(variable_name, when)
        footer_text = self._annotator.footer_for_static()
        return self._renderer.panels_static(
            arrays,
            cmap=cmaps,
            contour_arrays=contour_arrays,
            contour_level=self.plot_spec.ice_edge_threshold,
            dpi=self.plot_spec.dpi,
            figure_title=suptitle,
            footer_text=footer_text or None,
            group_axes=(0, 1)
            if self.plot_spec.include_difference or uncertainty is not None
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
        """Render a three-panel video BytesIO via Renderer.panels_video()."""
        masked_ground_truth = self.land_mask.apply_to(ground_truth)
        masked_prediction = self.land_mask.apply_to(prediction)

        arrays = [masked_ground_truth, masked_prediction]
        titles = [self.plot_spec.title_groundtruth, self.plot_spec.title_prediction]
        cmaps: list[str] = [self.plot_spec.colourmap, self.plot_spec.colourmap]
        vmins: list[float | None] = [self.plot_spec.vmin, self.plot_spec.vmin]
        vmaxs: list[float | None] = [self.plot_spec.vmax, self.plot_spec.vmax]

        if self.plot_spec.include_difference:
            difference, title, cmap, diff_vmin, diff_vmax = self._difference_panel(
                masked_ground_truth, masked_prediction
            )
            arrays.append(difference)
            titles.append(title)
            cmaps.append(cmap)
            vmins.append(diff_vmin)
            vmaxs.append(diff_vmax)

        contour_arrays: list[np.ndarray | None] | None = None
        if self.plot_spec.include_ice_edge:
            contour_arrays = [masked_ground_truth, masked_prediction]
            contour_arrays += [None] * (len(arrays) - len(contour_arrays))

        title_line = self._annotator.title_for_video(variable_name, dates, 0)
        footer_text = self._annotator.footer_for_video(dates)

        return self._renderer.panels_video(
            arrays,
            cmap=cmaps,
            contour_arrays=contour_arrays,
            contour_level=self.plot_spec.ice_edge_threshold,
            dpi=self.plot_spec.dpi,
            figure_title=title_line,
            footer_text=footer_text or None,
            fps=self.plot_spec.video_fps,
            group_axes=(0, 1) if self.plot_spec.include_difference else None,
            panel_titles=titles,
            vmax=vmaxs,
            vmin=vmins,
            video_format=self.plot_spec.video_format,
        )
