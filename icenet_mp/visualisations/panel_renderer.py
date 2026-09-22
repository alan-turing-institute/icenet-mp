from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import ArrayHW, ArrayTHW, Metadata, PlotSpec

from .colour_scale import ColourScale
from .difference_calculator import DifferenceCalculator
from .land_mask import LandMask
from .matplotlib_renderer import MatplotlibRenderer
from .media_annotator import MediaAnnotator
from .style_resolver import StyleResolver

if TYPE_CHECKING:
    from matplotlib.colors import Normalize

_VIDEO_NDIM = 3


class PanelRenderer:
    """Renders styled, land-masked panels for one land_mask/plot_spec pairing."""

    def __init__(
        self, land_mask: LandMask, metadata: Metadata, plot_spec: PlotSpec
    ) -> None:
        """Build a PanelRenderer for a given land mask, metadata, and plot spec."""
        self.land_mask = land_mask
        self.plot_spec = plot_spec
        self.annotator = MediaAnnotator(metadata, plot_spec)
        self.colour_scale = ColourScale(plot_spec.diff_mode)
        self.difference_calculator = DifferenceCalculator(plot_spec.diff_mode)
        self.renderer = MatplotlibRenderer()
        self.resolver = StyleResolver(
            plot_spec.per_variable_styles, plot_spec.colourmap
        )

    @property
    def video_format(self) -> Literal["mp4", "gif"]:
        return self.plot_spec.video_format

    def _get_difference(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        uncertainty: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Compute a difference between input arrays or return None if not requested."""
        if not self.plot_spec.include_difference:
            return None

        # If we have uncertainty data then calculate z-score
        if uncertainty is not None:
            return self.difference_calculator.standardised_difference(
                ground_truth, prediction, uncertainty
            )

        # Otherwise return the signed difference
        return self.difference_calculator.difference(ground_truth, prediction)

    def _validate_video_frames(
        self, arrays: list[ArrayTHW], dates: list[datetime]
    ) -> None:
        """Validate that video inputs are 3D [T, H, W] arrays matching `dates`.

        Raises:
            InvalidArrayError: If an array isn't 3D, or its frame count does not match
                the number of dates.

        """
        # Validate that the first array is 3D and has the correct number of frames
        shape = arrays[0].shape
        if len(shape) != _VIDEO_NDIM or shape[0] != len(dates):
            msg = (
                f"Expected a 3D [T, H, W] array with {len(dates)} frames, got {shape}."
            )
            raise InvalidArrayError(msg)
        # The remaining arrays only need to match the shape of the first array
        for array in arrays[1:]:
            if array.shape != shape:
                msg = f"Array shapes must match; expected {shape}, got {array.shape}."
                raise InvalidArrayError(msg)

    def static_singlet(
        self,
        values: ArrayHW,
        *,
        when: datetime,
        variable_name: str,
    ) -> ImageFile:
        """Render a single panel ImageFile via MatplotlibRenderer.panels_static().

        Args:
            values: 2D array of the variable field to render.
            when: Datetime of the plotted timestep.
            variable_name: Name of the variable being plotted, used for styling and
                title generation.

        Returns:
            An ImageFile containing the rendered panel.

        """
        masked_values = self.land_mask.apply_to(values)
        style = self.resolver.style_for_variable(variable_name)
        title = self.annotator.title_for_variable(variable_name, when, style.units)
        return self.renderer.panels_static(
            [masked_values],
            cmap=style.cmap,
            dpi=self.plot_spec.dpi,
            figure_title=title,
            vmax=style.vmax,
            vmin=style.vmin,
        )

    def static_triplet(
        self,
        ground_truth: ArrayHW,
        prediction: ArrayHW,
        *,
        panel_titles: dict[str, str] | None = None,
        uncertainty: ArrayHW | None = None,
        variable_name: str,
        when: datetime,
    ) -> ImageFile:
        """Render a three panel ImageFile via MatplotlibRenderer.panels().

        Args:
            ground_truth: 2D array of the ground truth field.
            prediction: 2D array of the predicted field.
            when: Datetime for the data in `ground_truth` and `prediction`.
            uncertainty: Optional 2D array of the reported standard uncertainty of the
                prediction field. When given, the third panel shows the standardised
                difference `z = (ground_truth - prediction) / uncertainty`.
            panel_titles: Optional overrides for the panel titles, keyed by
                "ground_truth", "prediction" and/or "difference".
            variable_name: Name of the variable being plotted, used for styling and
                title generation.

        Returns:
            An ImageFile containing the rendered panels.

        """
        masked_ground_truth = self.land_mask.apply_to(ground_truth)
        masked_prediction = self.land_mask.apply_to(prediction)
        panel_titles = panel_titles or {}

        arrays = [masked_ground_truth, masked_prediction]
        titles = [
            panel_titles.get("ground_truth", self.plot_spec.title_groundtruth),
            panel_titles.get("prediction", self.plot_spec.title_prediction),
        ]

        cmaps: list[str] = [
            self.plot_spec.colourmap,
            self.plot_spec.colourmap,
        ]
        norms: list[Normalize | None] = [None, None]
        vmins: list[float | None] = [self.plot_spec.vmin, self.plot_spec.vmin]
        vmaxs: list[float | None] = [self.plot_spec.vmax, self.plot_spec.vmax]

        # Optionally add a difference panel
        if (
            difference := self._get_difference(
                masked_ground_truth, masked_prediction, uncertainty
            )
        ) is not None:
            arrays.append(difference)
            titles.append(
                panel_titles.get(
                    "difference",
                    f"{self.plot_spec.title_difference} ({self.plot_spec.diff_mode})",
                )
            )
            diff_colour_scale = self.colour_scale.diff_colourmap(difference)
            norms.append(diff_colour_scale.norm)
            cmaps.append(diff_colour_scale.cmap)
            vmins.append(diff_colour_scale.bounds()[0])
            vmaxs.append(diff_colour_scale.bounds()[1])

        contour_arrays: list[np.ndarray | None] | None = None
        if self.plot_spec.include_ice_edge:
            contour_arrays = [masked_ground_truth, masked_prediction]
            contour_arrays += [None] * (len(arrays) - len(contour_arrays))

        title = self.annotator.title_for_static(variable_name, when)
        footer = self.annotator.footer_for_static()
        return self.renderer.panels_static(
            arrays,
            cmap=cmaps,
            contour_arrays=contour_arrays,
            contour_level=self.plot_spec.ice_edge_threshold,
            dpi=self.plot_spec.dpi,
            figure_title=title,
            footer_text=footer or None,
            group_axes=(0, 1)
            if self.plot_spec.include_difference or uncertainty is not None
            else None,
            norm=norms,
            panel_titles=titles,
            vmax=vmaxs,
            vmin=vmins,
        )

    def video_singlet(
        self,
        values: ArrayTHW,
        *,
        dates: list[datetime],
        variable_name: str,
    ) -> BytesIO:
        """Render a single panel video BytesIO via MatplotlibRenderer.panels_video().\

        Args:
            values: 3D array of the variable field to render, with shape (time, height, width).
            dates: List of datetimes corresponding to each timestep in `values`.
            variable_name: Name of the variable being plotted, used for styling and
                title generation.

        Returns:
            A BytesIO object containing the rendered video.

        Raises:
            InvalidArrayError: If `values` isn't 3D, or `dates` doesn't have
                one entry per frame.

        """
        self._validate_video_frames([values], dates)
        masked_values = self.land_mask.apply_to(values)
        style = self.resolver.style_for_variable(variable_name)

        def title_for_frame(tt: int) -> str:
            return self.annotator.title_for_variable(
                variable_name, dates[tt], style.units
            )

        return self.renderer.panels_video(
            [masked_values],
            cmap=style.cmap,
            dpi=self.plot_spec.dpi,
            figure_title=title_for_frame,
            fps=self.plot_spec.video_fps,
            vmax=style.vmax,
            vmin=style.vmin,
            video_format=self.video_format,
        )

    def video_triplet(
        self,
        ground_truth: ArrayTHW,
        prediction: ArrayTHW,
        *,
        dates: list[datetime],
        panel_titles: dict[str, str] | None = None,
        uncertainty: ArrayTHW | None = None,
        variable_name: str,
    ) -> BytesIO:
        """Render a three-panel video BytesIO via MatplotlibRenderer.panels_video().

        Args:
            ground_truth: 3D array of the ground truth field.
            prediction: 3D array of the predicted field.
            dates: Datetimes for the data in `ground_truth` and `prediction`.
            uncertainty: Optional 3D array of the reported standard uncertainty of the
                prediction field. When given, the third panel shows the standardised
                difference `z = (ground_truth - prediction) / uncertainty`.
            panel_titles: Optional overrides for the panel titles, keyed by
                "ground_truth", "prediction" and/or "difference".
            variable_name: Name of the variable being plotted, used for styling and
                title generation.

        Returns:
            A BytesIO object containing the rendered video.

        Raises:
            InvalidArrayError: If `ground_truth` or `prediction` isn't 3D, or
                `dates` doesn't have one entry per frame.

        """
        self._validate_video_frames([ground_truth, prediction], dates)
        masked_ground_truth = self.land_mask.apply_to(ground_truth)
        masked_prediction = self.land_mask.apply_to(prediction)
        panel_titles = panel_titles or {}

        arrays = [masked_ground_truth, masked_prediction]
        titles = [
            panel_titles.get("ground_truth", self.plot_spec.title_groundtruth),
            panel_titles.get("prediction", self.plot_spec.title_prediction),
        ]

        cmaps: list[str] = [
            self.plot_spec.colourmap,
            self.plot_spec.colourmap,
        ]
        vmins: list[float | None] = [self.plot_spec.vmin, self.plot_spec.vmin]
        vmaxs: list[float | None] = [self.plot_spec.vmax, self.plot_spec.vmax]

        # Optionally add a difference panel
        if (
            difference := self._get_difference(
                masked_ground_truth, masked_prediction, uncertainty
            )
        ) is not None:
            arrays.append(difference)
            titles.append(
                panel_titles.get(
                    "difference",
                    f"{self.plot_spec.title_difference} ({self.plot_spec.diff_mode})",
                )
            )
            diff_colour_scale = self.colour_scale.diff_colourmap(difference)
            cmaps.append(diff_colour_scale.cmap)
            vmins.append(diff_colour_scale.bounds()[0])
            vmaxs.append(diff_colour_scale.bounds()[1])

        contour_arrays: list[np.ndarray | None] | None = None
        if self.plot_spec.include_ice_edge:
            contour_arrays = [masked_ground_truth, masked_prediction]
            contour_arrays += [None] * (len(arrays) - len(contour_arrays))

        def title_for_frame(tt: int) -> str:
            return self.annotator.title_for_video(variable_name, dates, tt)

        footer = self.annotator.footer_for_video(dates)

        return self.renderer.panels_video(
            arrays,
            cmap=cmaps,
            contour_arrays=contour_arrays,
            contour_level=self.plot_spec.ice_edge_threshold,
            dpi=self.plot_spec.dpi,
            figure_title=title_for_frame,
            footer_text=footer or None,
            fps=self.plot_spec.video_fps,
            group_axes=(0, 1) if self.plot_spec.include_difference else None,
            panel_titles=titles,
            vmax=vmaxs,
            vmin=vmins,
            video_format=self.video_format,
        )
