import logging
from datetime import datetime

import numpy as np
from omegaconf import DictConfig

from icenet_mp.data import SingleDataset
from icenet_mp.exceptions import InvalidArrayError, VideoRenderError
from icenet_mp.types import (
    ArrayHW,
    ArrayTHW,
    Hemisphere,
    Metadata,
    ModelStepOutput,
    PlotSpec,
)

from .land_mask import LandMask
from .metadata import build_metadata, format_metadata_subtitle
from .plotting_static import plot_static_inputs, plot_static_prediction
from .plotting_timeseries import plot_time_trace
from .plotting_video import plot_video_inputs, plot_video_prediction

logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, plot_spec: PlotSpec) -> None:
        """A helper class to create and log plots."""
        self.plot_spec = plot_spec
        self.land_mask = LandMask(None)

    def get_metadata(self, config: DictConfig, model_name: str) -> Metadata:
        """Get metadata for the plotter based on the model test output."""
        return build_metadata(config, model_name)

    def set_metadata(self, metadata: Metadata) -> None:
        """Set metadata for the plotter, which may be used in titles and subtitles."""
        self.plot_spec.metadata_subtitle = format_metadata_subtitle(metadata)

    def log_static_inputs(
        self,
        inputs: list[SingleDataset],
        dates: list[datetime],
        image_loggers: list,
        prefix: str | None = None,
    ) -> None:
        """Extract and log static raw input plots."""
        try:
            idx_date = self.plot_spec.selected_timestep
            log_path = f"{prefix}/input_static" if prefix else "input_static"
            for input_ds in inputs:
                # Get static data for this timestep
                variables = {
                    f"{input_ds.name}:{v_name}": input_ds[idx_date][channel, :]
                    for channel, v_name in enumerate(input_ds.variable_names)
                }
                # Plot and log input static images
                images = plot_static_inputs(
                    variables,
                    land_mask=self.land_mask,
                    plot_spec=self.plot_spec,
                    when=dates[idx_date],
                )
                for image_name, image_list in images.items():
                    for image_logger in image_loggers:
                        image_logger.log_image(
                            key=f"{log_path}/{image_name}", images=image_list
                        )
        except InvalidArrayError as exc:
            logger.warning("Static plotting skipped due to invalid arrays: %s", exc)
        except (IndexError, ValueError, MemoryError, OSError) as exc:
            logger.warning("Static plotting failed: %s", exc)

    def log_static_outputs(
        self,
        outputs: ModelStepOutput,
        dates: list[datetime],
        image_loggers: list,
        channel_names: list[str],
        prefix: str | None = None,
    ) -> None:
        """Create and log static output plots."""
        try:
            idx_date = self.plot_spec.selected_timestep
            log_path = f"{prefix}/output_static" if prefix else "output_static"
            # Use all channels from the first batch -> [H,W]
            for idx_channel in range(outputs.target.shape[2]):
                ground_truth: ArrayHW = (
                    outputs.target[0, idx_date, idx_channel].detach().cpu().numpy()
                )
                prediction: ArrayHW = (
                    outputs.prediction[0, idx_date, idx_channel].detach().cpu().numpy()
                )
                variable_name = (
                    channel_names[idx_channel]
                    if idx_channel < len(channel_names)
                    else f"channel_{idx_channel}"
                )
                # Plot and log output static images
                images = plot_static_prediction(
                    ground_truth,
                    prediction,
                    date=dates[idx_date],
                    land_mask=self.land_mask,
                    plot_spec=self.plot_spec,
                    variable_name=variable_name,
                )
                for image_name, image_list in images.items():
                    for image_logger in image_loggers:
                        image_logger.log_image(
                            key=f"{log_path}/{image_name}", images=image_list
                        )
        except InvalidArrayError as err:
            logger.warning("Static plotting skipped due to invalid arrays: %s", err)
        except (IndexError, ValueError, MemoryError, OSError) as exc:
            logger.warning("Static plotting failed: %s", exc)

    def log_time_trace_outputs(
        self,
        outputs: ModelStepOutput,
        dates: list[datetime],
        image_loggers: list,
        channel_names: list[str],
        prefix: str | None = None,
    ) -> None:
        """Create and log prediction-vs-ground-truth time traces."""
        try:
            log_path = f"{prefix}/output_time_trace" if prefix else "output_time_trace"
            for idx_channel in range(outputs.target.shape[2]):
                ground_truth: ArrayTHW = (
                    outputs.target[0, :, idx_channel].detach().cpu().numpy()
                )
                prediction: ArrayTHW = (
                    outputs.prediction[0, :, idx_channel].detach().cpu().numpy()
                )
                variable_name = (
                    channel_names[idx_channel]
                    if idx_channel < len(channel_names)
                    else f"channel_{idx_channel}"
                )
                images = plot_time_trace(
                    ground_truth,
                    prediction,
                    dates=dates,
                    land_mask=self.land_mask,
                    variable_name=variable_name,
                    dpi=self.plot_spec.dpi,
                )
                for image_name, image_list in images.items():
                    for image_logger in image_loggers:
                        image_logger.log_image(
                            key=f"{log_path}/{image_name}", images=image_list
                        )
        except InvalidArrayError as exc:
            logger.warning("Time-trace plotting skipped due to invalid arrays: %s", exc)
        except (IndexError, ValueError, MemoryError, OSError) as exc:
            logger.warning("Time-trace plotting failed: %s", exc)

    def log_video_inputs(
        self,
        inputs: list[SingleDataset],
        dates: list[datetime],
        video_loggers: list,
        prefix: str | None = None,
    ) -> None:
        """Extract and log raw input videos."""
        log_path = f"{prefix}/input_video" if prefix else "input_video"
        for input_ds in inputs:
            # Create animations for all variables
            np_dates = [np.datetime64(date.replace(tzinfo=None)) for date in dates]
            variables = {
                f"{input_ds.name}:{v_name}": input_ds.get_tchw(np_dates)[:, channel, :]
                for channel, v_name in enumerate(input_ds.variable_names)
            }
            videos = plot_video_inputs(
                variables,
                dates=dates,
                plot_spec=self.plot_spec,
                land_mask=self.land_mask,
            )

            # Log input videos
            for video_logger in video_loggers:
                for video_name, video_buffer in videos.items():
                    video_buffer.seek(0)
                    video_logger.log_video(
                        key=f"{log_path}/{video_name}",
                        videos=[video_buffer],
                        format=[self.plot_spec.video_format],
                    )

    def log_video_outputs(
        self,
        outputs: ModelStepOutput,
        dates: list[datetime],
        video_loggers: list,
        channel_names: list[str],
        prefix: str | None = None,
    ) -> None:
        """Create and log output videos."""
        try:
            log_path = f"{prefix}/output_video" if prefix else "output_video"
            # Use all channels from the first batch -> [T, H,W]
            for idx_channel in range(outputs.target.shape[2]):
                ground_truth: ArrayTHW = (
                    outputs.target[0, :, idx_channel].detach().cpu().numpy()
                )
                prediction: ArrayTHW = (
                    outputs.prediction[0, :, idx_channel].detach().cpu().numpy()
                )
                variable_name = (
                    channel_names[idx_channel]
                    if idx_channel < len(channel_names)
                    else f"channel_{idx_channel}"
                )
                video_data = plot_video_prediction(
                    ground_truth,
                    prediction,
                    dates=dates,
                    land_mask=self.land_mask,
                    plot_spec=self.plot_spec,
                    variable_name=variable_name,
                )
                for video_logger in video_loggers:
                    for video_name, video_buffer in video_data.items():
                        video_buffer.seek(0)
                        video_logger.log_video(
                            key=f"{log_path}/{video_name}",
                            videos=[video_buffer],
                            format=[self.plot_spec.video_format],
                        )
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
