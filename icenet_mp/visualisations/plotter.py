import logging
from datetime import datetime
from io import BytesIO

from omegaconf import DictConfig
from PIL.ImageFile import ImageFile

from icenet_mp.data import SingleDataset
from icenet_mp.exceptions import InvalidArrayError, VideoRenderError
from icenet_mp.types import (
    ArrayHW,
    ArrayTHW,
    Hemisphere,
    Metadata,
    ModelStepOutput,
    PlotSpec,
    SupportsImageLogging,
    SupportsVideoLogging,
    UncertaintyArrays,
)
from icenet_mp.utils import npdatetime_from_datetime

from .difference_calculator import DifferenceCalculator
from .land_mask import LandMask
from .metadata_builder import MetadataBuilder
from .plot_annotator import PlotAnnotator
from .plotting_static import plot_static_inputs, plot_static_uncertainty
from .plotting_video import plot_video_inputs
from .render import render_panels_static, render_panels_video

logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, plot_spec: PlotSpec | None = None) -> None:
        """A helper class to create and log plots."""
        self.plot_spec = plot_spec if plot_spec is not None else PlotSpec()
        self.land_mask = LandMask(None)
        self.metadata_builder = MetadataBuilder()

    @staticmethod
    def _log_path(prefix: str | None, name: str) -> str:
        """Build a consistent logger namespace."""
        return f"{prefix}/{name}" if prefix else name

    @staticmethod
    def _channel_name(channel_names: list[str], idx_channel: int) -> str:
        """Return the configured channel name or a stable fallback."""
        if idx_channel < len(channel_names):
            return channel_names[idx_channel]
        return f"channel_{idx_channel}"

    @staticmethod
    def _log_images(
        images: dict[str, list[ImageFile]],
        image_loggers: list[SupportsImageLogging],
        log_path: str,
    ) -> None:
        """Send rendered image groups to every configured image logger."""
        for image_name, image_list in images.items():
            for image_logger in image_loggers:
                image_logger.log_image(
                    key=f"{log_path}/{image_name}", images=image_list
                )

    def _log_videos(
        self,
        videos: dict[str, BytesIO],
        video_loggers: list[SupportsVideoLogging],
        log_path: str,
    ) -> None:
        """Rewind and send rendered videos to every configured video logger."""
        for video_logger in video_loggers:
            for video_name, video_buffer in videos.items():
                video_buffer.seek(0)
                video_logger.log_video(
                    key=f"{log_path}/{video_name}",
                    videos=[video_buffer],
                    format=[self.plot_spec.video_format],
                )

    def get_metadata(self, config: DictConfig, model_name: str) -> Metadata:
        """Get metadata for the plotter based on the model test output."""
        return self.metadata_builder.build(config, model_name)

    def set_metadata(self, metadata: Metadata) -> None:
        """Set metadata for the plotter, which may be used in titles and subtitles."""
        self.plot_spec.metadata_subtitle = self.metadata_builder.format_subtitle(
            metadata
        )

    def log_static_inputs(
        self,
        inputs: list[SingleDataset],
        dates: list[datetime],
        image_loggers: list[SupportsImageLogging],
        prefix: str | None = None,
    ) -> None:
        """Extract and log static raw input plots."""
        try:
            idx_date = self.plot_spec.selected_timestep
            log_path = self._log_path(prefix, "input_static")
            for input_ds in inputs:
                # Get data for all variables at the selected timestep
                variables = {
                    f"{input_ds.name}:{v_name}": input_ds[idx_date][channel, :]
                    for channel, v_name in enumerate(input_ds.variable_names)
                }
                # Plot static input images
                images = plot_static_inputs(
                    variables,
                    land_mask=self.land_mask,
                    plot_spec=self.plot_spec,
                    when=dates[idx_date],
                )
                # Log static input images
                self._log_images(images, image_loggers, log_path)
        except InvalidArrayError as exc:
            logger.warning("Static plotting skipped due to invalid arrays: %s", exc)
        except (IndexError, ValueError, MemoryError, OSError) as exc:
            logger.warning("Static plotting failed: %s", exc)

    def _render_static_prediction(
        self,
        ground_truth: ArrayHW,
        prediction: ArrayHW,
        *,
        when: datetime,
        variable_name: str,
    ) -> ImageFile:
        """Render the ground-truth/prediction(/difference) triptych via render_panels."""
        plot_spec = self.plot_spec
        masked_ground_truth = self.land_mask.apply_to(ground_truth)
        masked_prediction = self.land_mask.apply_to(prediction)

        arrays = [masked_ground_truth, masked_prediction]
        titles = [plot_spec.title_groundtruth, plot_spec.title_prediction]
        cmaps: list[str] = [plot_spec.colourmap, plot_spec.colourmap]
        vmins: list[float | None] = [plot_spec.vmin, plot_spec.vmin]
        vmaxs: list[float | None] = [plot_spec.vmax, plot_spec.vmax]

        if plot_spec.include_difference:
            difference_calculator = DifferenceCalculator()
            difference = self.land_mask.apply_to(
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
        suptitle = annotator.title_for_static(variable_name, plot_spec, when)
        footer_text = annotator.footer_for_static(plot_spec)
        return render_panels_static(
            arrays,
            cmap=cmaps,
            dpi=plot_spec.dpi,
            figure_title=suptitle,
            footer_text=footer_text or None,
            group_axes=(0, 1) if plot_spec.include_difference else None,
            panel_titles=titles,
            vmax=vmaxs,
            vmin=vmins,
        )

    def log_static_outputs(
        self,
        outputs: ModelStepOutput,
        dates: list[datetime],
        image_loggers: list[SupportsImageLogging],
        channel_names: list[str],
        prefix: str | None = None,
        uncertainties: dict[int, ArrayTHW] | None = None,
    ) -> None:
        """Create and log static output plots, including uncertainty when available."""
        try:
            idx_date = self.plot_spec.selected_timestep
            log_path = self._log_path(prefix, "output_static")
            # Use all channels from the first batch -> [H,W]
            for idx_channel in range(outputs.target.shape[2]):
                ground_truth: ArrayHW = (
                    outputs.target[0, idx_date, idx_channel].detach().cpu().numpy()
                )
                prediction: ArrayHW = (
                    outputs.prediction[0, idx_date, idx_channel].detach().cpu().numpy()
                )
                variable_name = self._channel_name(channel_names, idx_channel)
                # Plot static prediction image via the minimal render_panels core
                image = self._render_static_prediction(
                    ground_truth,
                    prediction,
                    when=dates[idx_date],
                    variable_name=variable_name,
                )
                date_key = dates[idx_date].strftime(r"%Y-%m-%d")
                images: dict[str, list[ImageFile]] = {
                    f"{date_key}-{variable_name}": [image]
                }
                # Plot static uncertainty images
                if (
                    uncertainty := (
                        uncertainties.get(idx_channel)
                        if uncertainties is not None
                        else None
                    )
                ) is not None:
                    images.update(
                        plot_static_uncertainty(
                            UncertaintyArrays(
                                ground_truth, prediction, uncertainty[idx_date]
                            ),
                            date=dates[idx_date],
                            land_mask=self.land_mask,
                            plot_spec=self.plot_spec,
                            variable_name=variable_name,
                        )
                    )
                # Log static output images
                self._log_images(images, image_loggers, log_path)
        except InvalidArrayError as err:
            logger.warning("Static plotting skipped due to invalid arrays: %s", err)
        except (IndexError, ValueError, MemoryError, OSError) as exc:
            logger.warning("Static plotting failed: %s", exc)

    def log_video_inputs(
        self,
        inputs: list[SingleDataset],
        dates: list[datetime],
        video_loggers: list[SupportsVideoLogging],
        prefix: str | None = None,
    ) -> None:
        """Extract and log raw input videos."""
        try:
            log_path = self._log_path(prefix, "input_video")
            for input_ds in inputs:
                # Get data for all variables at the selected timestep
                np_dates = [npdatetime_from_datetime(date) for date in dates]
                variables = {
                    f"{input_ds.name}:{v_name}": input_ds.get_tchw(np_dates)[
                        :, channel, :
                    ]
                    for channel, v_name in enumerate(input_ds.variable_names)
                }
                # Plot input animations
                videos = plot_video_inputs(
                    variables,
                    dates=dates,
                    plot_spec=self.plot_spec,
                    land_mask=self.land_mask,
                )
                # Log input animations
                self._log_videos(videos, video_loggers, log_path)
        except (InvalidArrayError, VideoRenderError) as err:
            logger.warning("Video plotting skipped: %s", err)
        except (IndexError, ValueError, MemoryError, OSError):
            logger.exception("Video plotting failed")

    def _render_video_prediction(
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
            difference_calculator = DifferenceCalculator()
            difference = self.land_mask.apply_to(
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

    def log_video_outputs(
        self,
        outputs: ModelStepOutput,
        dates: list[datetime],
        video_loggers: list[SupportsVideoLogging],
        channel_names: list[str],
        prefix: str | None = None,
    ) -> None:
        """Create and log output videos."""
        try:
            log_path = self._log_path(prefix, "output_video")
            # Use all channels from the first batch -> [H,W]
            for idx_channel in range(outputs.target.shape[2]):
                ground_truth: ArrayTHW = (
                    outputs.target[0, :, idx_channel].detach().cpu().numpy()
                )
                prediction: ArrayTHW = (
                    outputs.prediction[0, :, idx_channel].detach().cpu().numpy()
                )
                variable_name = self._channel_name(channel_names, idx_channel)
                # Plot output animation via the minimal render_panels core
                video = self._render_video_prediction(
                    ground_truth,
                    prediction,
                    dates=dates,
                    variable_name=variable_name,
                )
                date_key = dates[0].strftime(r"%Y-%m-%d")
                video_data = {f"{date_key}-{variable_name}": video}
                # Log output animations
                self._log_videos(video_data, video_loggers, log_path)
        except (InvalidArrayError, VideoRenderError) as err:
            logger.warning("Video plotting skipped: %s", err)
        except (IndexError, ValueError, MemoryError, OSError):
            logger.exception("Video plotting failed")

    def set_hemisphere(
        self,
        hemisphere: Hemisphere,
    ) -> None:
        """Set the hemisphere and update the plot spec accordingly."""
        self.plot_spec.hemisphere = hemisphere
