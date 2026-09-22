import logging
from contextlib import suppress
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL.ImageFile import ImageFile

from icenet_mp.data import CombinedDataset, SingleDataset
from icenet_mp.exceptions import InvalidArrayError, VideoRenderError
from icenet_mp.types import (
    ArrayHW,
    ArrayTCHW,
    ArrayTHW,
    Metadata,
    ModelStepOutput,
    PlotSpec,
    SupportsImageLogging,
    SupportsVideoLogging,
)
from icenet_mp.utils import npdatetime_from_datetime

from .land_mask import LandMask
from .panel_renderer import PanelRenderer

logger = logging.getLogger(__name__)


class MediaPublisher:
    """Publish static and video plots for a dataset/plot_spec/land_mask context."""

    def __init__(
        self,
        *,
        dataset: CombinedDataset,
        plot_spec: PlotSpec,
        land_mask: LandMask,
        current_epoch: int | None = None,
        model_name: str | None = None,
    ) -> None:
        """Build a publisher bound to one dataset/plot_spec/land_mask context."""
        self._plot_spec = plot_spec
        self._panel_renderer = PanelRenderer(
            land_mask,
            self.build_metadata(dataset, current_epoch, model_name),
            plot_spec,
        )

    @staticmethod
    def build_metadata(
        dataset: CombinedDataset,
        current_epoch: int | None = None,
        model_name: str | None = None,
    ) -> Metadata:
        """Build structured metadata from a CombinedDataset."""
        # Format the dataset's frequency as a short, human-readable cadence label.
        hours = float(dataset.frequency / np.timedelta64(1, "h"))
        if hours % 24 == 0:
            days = int(hours // 24)
            cadence = "daily" if days == 1 else f"{days}d"
        else:
            cadence = "hourly" if hours == 1 else f"{hours:g}h"

        vars_by_source = {ds.name: sorted(ds.variable_names) for ds in dataset.inputs}

        return Metadata(
            model=model_name,
            current_epoch=current_epoch,
            start=str(dataset.start_date.astype("datetime64[D]")),
            end=str(dataset.end_date.astype("datetime64[D]")),
            cadence=cadence,
            n_points=len(dataset),
            n_history_steps=dataset.n_history_steps,
            vars_by_source=vars_by_source or None,
        )

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

    @staticmethod
    def _log_path(prefix: str | None, name: str) -> str:
        """Build a consistent logger namespace."""
        return f"{prefix}/{name}" if prefix else name

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
                    format=[self._plot_spec.video_format],
                )

    def log_static_inputs(
        self,
        inputs: list[SingleDataset],
        dates: list[datetime],
        image_loggers: list[SupportsImageLogging],
        *,
        prefix: str | None = None,
    ) -> None:
        """Extract and log static raw input plots."""
        try:
            idx_date = self._plot_spec.selected_timestep
            when = dates[idx_date]
            log_path = self._log_path(prefix, "input_static")
            for input_ds in inputs:
                # Get data for all variables at the selected timestep
                for channel, v_name in enumerate(input_ds.variable_names):
                    variable_name = f"{input_ds.name}:{v_name}"
                    image = self._panel_renderer.static_singlet(
                        input_ds[idx_date][channel, :],
                        when=when,
                        variable_name=variable_name,
                    )
                    key = f"{when.strftime(r'%Y-%m-%d')}-{variable_name}"
                    images: dict[str, list[ImageFile]] = {key: [image]}
                    # Log static input images
                    self._log_images(images, image_loggers, log_path)
        except InvalidArrayError as exc:
            logger.warning("Static plotting skipped due to invalid arrays: %s", exc)
        except (IndexError, ValueError, MemoryError, OSError) as exc:
            logger.warning("Static plotting failed: %s", exc)

    def log_static_outputs(  # noqa: PLR0913
        self,
        outputs: ModelStepOutput,
        dates: list[datetime],
        image_loggers: list[SupportsImageLogging],
        *,
        channel_names: list[str] | None = None,
        climatology: ArrayTCHW | None = None,
        prefix: str | None = None,
        uncertainties: dict[int, ArrayTHW] | None = None,
    ) -> None:
        """Create and log static output plots, including climatology when available.

        Also logs a standardised uncertainty plot and, when a climatology table is
        given, a calendar-day-mean (climatology) map for the plotted date and channel.
        """
        try:
            idx_date = self._plot_spec.selected_timestep
            log_path = self._log_path(prefix, "output_static")
            # Use all channels from the first batch -> [H,W]
            for idx_channel in range(outputs.target.shape[2]):
                ground_truth: ArrayHW = (
                    outputs.target[0, idx_date, idx_channel].detach().cpu().numpy()
                )
                prediction: ArrayHW = (
                    outputs.prediction[0, idx_date, idx_channel].detach().cpu().numpy()
                )
                variable_name = self._channel_name(channel_names or [], idx_channel)
                date_key = dates[idx_date].strftime(r"%Y-%m-%d")
                images: dict[str, list[ImageFile]] = {}
                # Plot static truth/prediction/difference image
                images[f"{date_key}-{variable_name}-truth-difference"] = [
                    self._panel_renderer.static_triplet(
                        ground_truth,
                        prediction,
                        when=dates[idx_date],
                        variable_name=variable_name,
                    )
                ]
                # Plot static climatology/prediction/difference image
                if climatology is not None:
                    with suppress(IndexError, TypeError):
                        climatology_field = climatology[idx_date, idx_channel]
                        images[f"{date_key}-{variable_name}-climatology-difference"] = [
                            self._panel_renderer.static_triplet(
                                climatology_field,
                                prediction,
                                panel_titles={"ground_truth": "Climatology"},
                                variable_name=variable_name,
                                when=dates[idx_date],
                            )
                        ]
                # Plot static truth/prediction/z-score image
                if (
                    uncertainty := (
                        uncertainties.get(idx_channel)
                        if uncertainties is not None
                        else None
                    )
                ) is not None:
                    images[f"{date_key}-{variable_name}-z-score"] = [
                        self._panel_renderer.static_triplet(
                            ground_truth,
                            prediction,
                            panel_titles={"difference": "Standardised Difference (z)"},
                            variable_name=variable_name,
                            uncertainty=uncertainty[idx_date],
                            when=dates[idx_date],
                        )
                    ]
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
        *,
        prefix: str | None = None,
    ) -> None:
        """Extract and log raw input videos."""
        try:
            log_path = self._log_path(prefix, "input_video")
            np_dates = [npdatetime_from_datetime(date) for date in dates]
            date_key = dates[0].strftime(r"%Y-%m-%d")
            for input_ds in inputs:
                # Get data for all variables over the full date range
                for channel, v_name in enumerate(input_ds.variable_names):
                    variable_name = f"{input_ds.name}:{v_name}"
                    video = self._panel_renderer.video_singlet(
                        input_ds.get_tchw(np_dates)[:, channel, :],
                        dates=dates,
                        variable_name=variable_name,
                    )
                    video_data = {f"{date_key}-{variable_name}": video}
                    # Log input animations
                    self._log_videos(video_data, video_loggers, log_path)
        except (InvalidArrayError, VideoRenderError) as err:
            logger.warning("Video plotting skipped: %s", err)
        except (IndexError, ValueError, MemoryError, OSError):
            logger.exception("Video plotting failed")

    def log_video_outputs(  # noqa: PLR0913
        self,
        outputs: ModelStepOutput,
        dates: list[datetime],
        video_loggers: list[SupportsVideoLogging],
        *,
        channel_names: list[str] | None = None,
        climatology: ArrayTCHW | None = None,
        prefix: str | None = None,
        uncertainties: dict[int, ArrayTHW] | None = None,
    ) -> None:
        """Create and log output videos."""
        try:
            log_path = self._log_path(prefix, "output_video")
            date_key = dates[0].strftime(r"%Y-%m-%d")
            videos: dict[str, BytesIO] = {}
            # Use all channels from the first batch -> [H,W]
            for idx_channel in range(outputs.target.shape[2]):
                ground_truth: ArrayTHW = (
                    outputs.target[0, :, idx_channel].detach().cpu().numpy()
                )
                prediction: ArrayTHW = (
                    outputs.prediction[0, :, idx_channel].detach().cpu().numpy()
                )
                variable_name = self._channel_name(channel_names or [], idx_channel)
                # Plot truth/prediction/difference video
                videos[f"{date_key}-{variable_name}-truth-difference"] = (
                    self._panel_renderer.video_triplet(
                        ground_truth,
                        prediction,
                        dates=dates,
                        variable_name=variable_name,
                    )
                )
                # Plot climatology/prediction/difference video
                if climatology is not None:
                    with suppress(IndexError, TypeError):
                        climatology_thw = climatology[:, idx_channel, :, :]
                        videos[f"{date_key}-{variable_name}-climatology-difference"] = (
                            self._panel_renderer.video_triplet(
                                climatology_thw,
                                prediction,
                                dates=dates,
                                panel_titles={"ground_truth": "Climatology"},
                                variable_name=variable_name,
                            )
                        )
                # Plot static truth/prediction/z-score image
                if (
                    uncertainty := (
                        uncertainties.get(idx_channel)
                        if uncertainties is not None
                        else None
                    )
                ) is not None:
                    videos[f"{date_key}-{variable_name}-z-score"] = (
                        self._panel_renderer.video_triplet(
                            ground_truth,
                            prediction,
                            dates=dates,
                            panel_titles={"difference": "Standardised Difference (z)"},
                            uncertainty=uncertainty,
                            variable_name=variable_name,
                        )
                    )
            # Log output animations
            self._log_videos(videos, video_loggers, log_path)
        except (InvalidArrayError, VideoRenderError) as err:
            logger.warning("Video plotting skipped: %s", err)
        except (IndexError, ValueError, MemoryError, OSError):
            logger.exception("Video plotting failed")
