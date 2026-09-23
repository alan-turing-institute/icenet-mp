import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from io import BytesIO
from typing import TypeVar

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
    Timespan,
)
from icenet_mp.utils import iso_from_date, npdatetime_from_datetime

from .land_mask import LandMask
from .panel_renderer import PanelRenderer

log = logging.getLogger(__name__)

RenderedMedia = TypeVar("RenderedMedia", ImageFile, BytesIO)


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
        self.idx_date = plot_spec.selected_timestep
        self.panel_renderer = PanelRenderer(
            land_mask,
            Metadata.from_dataset(
                dataset, current_epoch=current_epoch, model_name=model_name
            ),
            plot_spec,
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

    @staticmethod
    def _render_three_panel_media(
        *,
        climatology: np.ndarray | None,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        render: Callable[..., RenderedMedia],
        uncertainty: np.ndarray | None,
        variable_name: str,
        **render_kwargs: object,
    ) -> dict[str, RenderedMedia]:
        """Render the configured three-panel media.

        These may include truth/prediction/difference, climatology/prediction/difference
        and truth/prediction/z-score.

        Shared by the static and video loggers, which differ only in the `render`
        callable used and the extra `when`/`dates` keyword each one needs.
        """
        media: dict[str, RenderedMedia] = {
            "truth-difference": render(
                ground_truth, prediction, variable_name=variable_name, **render_kwargs
            )
        }
        if climatology is not None:
            with suppress(IndexError, TypeError):
                media["climatology-difference"] = render(
                    climatology,
                    prediction,
                    panel_titles={"ground_truth": "Climatology"},
                    variable_name=variable_name,
                    **render_kwargs,
                )
        if uncertainty is not None:
            media["z-score"] = render(
                ground_truth,
                prediction,
                panel_titles={"difference": "Standardised Difference (z)"},
                uncertainty=uncertainty,
                variable_name=variable_name,
                **render_kwargs,
            )
        return media

    @staticmethod
    def _select_climatology(
        climatology: ArrayTCHW | None,
        idx_channel: int,
        idx_date: int | None,
    ) -> np.ndarray | None:
        """Return the climatology array for a given channel and (optionally) date."""
        if climatology is None:
            return None
        with suppress(IndexError, TypeError):
            return climatology[
                slice(None) if idx_date is None else idx_date, idx_channel
            ]
        return None

    @staticmethod
    def _select_uncertainty(
        uncertainties: dict[int, ArrayTHW] | None,
        idx_channel: int,
        idx_date: int | None,
    ) -> np.ndarray | None:
        """Return the uncertainty array for a given channel and (optionally) date."""
        if (
            uncertainties is None
            or (uncertainty := uncertainties.get(idx_channel)) is None
        ):
            return None
        return uncertainty[idx_date] if idx_date is not None else uncertainty

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
                    format=[self.panel_renderer.video_format],
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
            when = dates[self.idx_date]
            log_path = self._log_path(prefix, "input_static")
            for input_ds in inputs:
                # Get data for all variables at the selected timestep
                for channel, v_name in enumerate(input_ds.variable_names):
                    variable_name = f"{input_ds.name}:{v_name}"
                    image = self.panel_renderer.static_singlet(
                        input_ds[self.idx_date][channel, :],
                        when=when,
                        variable_name=variable_name,
                    )
                    key = f"{iso_from_date(when)}-{variable_name}"
                    images: dict[str, list[ImageFile]] = {key: [image]}
                    # Log static input images
                    self._log_images(images, image_loggers, log_path)
        except (InvalidArrayError, IndexError, ValueError, MemoryError, OSError) as exc:
            log.warning("Image logging failed: %s", exc)

    def log_static_outputs(  # noqa: PLR0913
        self,
        outputs: ModelStepOutput,
        image_loggers: list[SupportsImageLogging],
        *,
        channel_names: list[str] | None = None,
        climatology: ArrayTCHW | None = None,
        forecast_dates: list[datetime],
        history_dates: list[datetime],
        prefix: str | None = None,
        uncertainties: dict[int, ArrayTHW] | None = None,
    ) -> None:
        """Create and log static output plots, including climatology when available.

        Also logs a standardised uncertainty plot and, when a climatology table is
        given, a calendar-day-mean (climatology) map for the plotted date and channel.
        """
        try:
            log_path = self._log_path(prefix, "output_static")
            date_key = iso_from_date(forecast_dates[self.idx_date])
            # Use all channels from the first batch -> [H,W]
            for idx_channel in range(outputs.target.shape[2]):
                ground_truth: ArrayHW = (
                    outputs.target[0, self.idx_date, idx_channel].detach().cpu().numpy()
                )
                prediction: ArrayHW = (
                    outputs.prediction[0, self.idx_date, idx_channel]
                    .detach()
                    .cpu()
                    .numpy()
                )
                variable_name = self._channel_name(channel_names or [], idx_channel)
                media = self._render_three_panel_media(
                    ground_truth=ground_truth,
                    prediction=prediction,
                    variable_name=variable_name,
                    render=self.panel_renderer.static_triplet,
                    climatology=self._select_climatology(
                        climatology, idx_channel, self.idx_date
                    ),
                    uncertainty=self._select_uncertainty(
                        uncertainties, idx_channel, self.idx_date
                    ),
                    history_ctx=Timespan(start=history_dates[0], end=history_dates[-1]),
                    forecast_date=forecast_dates[self.idx_date],
                )
                images: dict[str, list[ImageFile]] = {
                    f"{date_key}-{variable_name}-{suffix}": [image]
                    for suffix, image in media.items()
                }
                # Log static output images
                self._log_images(images, image_loggers, log_path)
        except (InvalidArrayError, IndexError, ValueError, MemoryError, OSError) as exc:
            log.warning("Image logging failed: %s", exc)

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
            date_key = iso_from_date(dates[0])
            for input_ds in inputs:
                # Get data for all variables over the full date range
                for channel, unqualified_name in enumerate(input_ds.variable_names):
                    variable_name = f"{input_ds.name}:{unqualified_name}"
                    video = self.panel_renderer.video_singlet(
                        input_ds.get_tchw(np_dates)[:, channel, :],
                        dates=dates,
                        variable_name=variable_name,
                    )
                    video_data = {f"{date_key}-{variable_name}": video}
                    # Log input animations
                    self._log_videos(video_data, video_loggers, log_path)
        except (
            IndexError,
            InvalidArrayError,
            MemoryError,
            OSError,
            ValueError,
            VideoRenderError,
        ) as exc:
            log.warning("Video logging failed: %s", exc)

    def log_video_outputs(  # noqa: PLR0913
        self,
        outputs: ModelStepOutput,
        video_loggers: list[SupportsVideoLogging],
        *,
        channel_names: list[str] | None = None,
        climatology: ArrayTCHW | None = None,
        forecast_dates: list[datetime],
        history_dates: list[datetime],
        prefix: str | None = None,
        uncertainties: dict[int, ArrayTHW] | None = None,
    ) -> None:
        """Create and log output videos."""
        try:
            log_path = self._log_path(prefix, "output_video")
            date_key = iso_from_date(forecast_dates[0])
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
                media = self._render_three_panel_media(
                    ground_truth=ground_truth,
                    prediction=prediction,
                    variable_name=variable_name,
                    render=self.panel_renderer.video_triplet,
                    climatology=self._select_climatology(
                        climatology, idx_channel, None
                    ),
                    uncertainty=self._select_uncertainty(
                        uncertainties, idx_channel, None
                    ),
                    history_ctx=Timespan(start=history_dates[0], end=history_dates[-1]),
                    forecast_ctx=Timespan(
                        start=forecast_dates[0], end=forecast_dates[-1]
                    ),
                )
                videos.update(
                    {
                        f"{date_key}-{variable_name}-{suffix}": video
                        for suffix, video in media.items()
                    }
                )
            # Log output animations
            self._log_videos(videos, video_loggers, log_path)
        except (
            IndexError,
            InvalidArrayError,
            MemoryError,
            OSError,
            ValueError,
            VideoRenderError,
        ) as exc:
            log.warning("Video logging failed: %s", exc)
