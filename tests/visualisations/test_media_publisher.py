import logging
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, ClassVar, cast
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch

from icenet_mp.data import SingleDataset
from icenet_mp.exceptions import InvalidArrayError, VideoRenderError
from icenet_mp.types import Metadata, ModelStepOutput, PlotSpec
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.media_publisher import MediaPublisher

if TYPE_CHECKING:
    from PIL.ImageFile import ImageFile

N_TIMESTEPS = 2
N_CHANNELS = 2
HEIGHT = 4
WIDTH = 4

TEST_DATES = [datetime(2020, 1, 1), datetime(2020, 1, 2)]


def fake_single_dataset() -> SingleDataset:
    """Return a duck-typed SingleDataset stand-in, cast to satisfy MediaPublisher's typing."""

    class FakeSingleDataset:
        """Minimal SingleDataset stand-in exposing the attributes MediaPublisher reads."""

        name = "example"
        variable_names: ClassVar[list[str]] = ["ice_conc", "temperature"]

        def __getitem__(self, _idx: int) -> np.ndarray:
            """Return a deterministic [C, H, W] array for the requested timestep."""
            return np.ones((N_CHANNELS, HEIGHT, WIDTH), dtype=np.float32)

        def get_tchw(self, dates: list) -> np.ndarray:
            """Return a deterministic [T, C, H, W] array for the requested dates."""
            return np.ones((len(dates), N_CHANNELS, HEIGHT, WIDTH), dtype=np.float32)

    return cast("SingleDataset", FakeSingleDataset())


def make_model_step_output(channels: int = N_CHANNELS) -> ModelStepOutput:
    """Build a ModelStepOutput with real tensors shaped [N, T, C, H, W]."""
    shape = (1, N_TIMESTEPS, channels, HEIGHT, WIDTH)
    return ModelStepOutput(
        prediction=torch.zeros(shape),
        target=torch.ones(shape),
        loss=torch.tensor(0.0),
    )


class TestLoggingHelpers:
    def test_log_path_handles_optional_prefix(self) -> None:
        """Build the same namespaces for prefixed and unprefixed logging."""
        assert MediaPublisher._log_path(None, "output_static") == "output_static"
        assert MediaPublisher._log_path("test", "output_static") == "test/output_static"

    def test_channel_name_uses_stable_fallback(self) -> None:
        """Use configured names when available and indexed fallbacks otherwise."""
        assert MediaPublisher._channel_name(["sic"], 0) == "sic"
        assert MediaPublisher._channel_name(["sic"], 2) == "channel_2"

    def test_log_images_fans_out_to_all_loggers(self) -> None:
        """Send every image group to each configured logger."""
        first = MagicMock()
        second = MagicMock()
        images = cast(
            "dict[str, list[ImageFile]]",
            {"comparison": [object()], "error": [object()]},
        )

        MediaPublisher._log_images(images, [first, second], "validation/output_static")

        expected = [
            {
                "key": "validation/output_static/comparison",
                "images": images["comparison"],
            },
            {"key": "validation/output_static/error", "images": images["error"]},
        ]
        assert [c.kwargs for c in first.log_image.call_args_list] == expected
        assert [c.kwargs for c in second.log_image.call_args_list] == expected

    def test_log_videos_rewinds_for_each_logger_and_preserves_format(self) -> None:
        """Rewind shared buffers before every logger handoff."""
        media_publisher = MediaPublisher(PlotSpec(video_format="mp4"))
        first = MagicMock()
        second = MagicMock()
        buffer = BytesIO(b"video")
        buffer.seek(3)

        media_publisher._log_videos(
            {"forecast": buffer},
            [first, second],
            "test/output_video",
        )

        first.log_video.assert_called_once_with(
            key="test/output_video/forecast", videos=[buffer], format=["mp4"]
        )
        second.log_video.assert_called_once_with(
            key="test/output_video/forecast", videos=[buffer], format=["mp4"]
        )
        assert buffer.tell() == 0


class TestMetadataAndHemisphere:
    def test_configure_context_updates_metadata_subtitle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Build metadata from the dataset and store its formatted subtitle."""
        media_publisher = MediaPublisher()
        monkeypatch.setattr(
            media_publisher._metadata_builder,
            "from_dataset",
            MagicMock(return_value=Metadata(model="unet")),
        )
        monkeypatch.setattr(
            media_publisher._annotator,
            "format_subtitle",
            MagicMock(return_value="epochs=50"),
        )

        media_publisher.configure_context(
            dataset=MagicMock(), current_epoch=50, model_name="unet"
        )

        assert media_publisher.plot_spec.metadata_subtitle == "epochs=50"

    def test_configure_context_updates_hemisphere(self) -> None:
        """MediaPublisher keeps hemisphere state on its PlotSpec."""
        media_publisher = MediaPublisher()
        media_publisher.configure_context(hemisphere="south")

        assert media_publisher.plot_spec.hemisphere == "south"

    def test_configure_context_updates_land_mask_and_renderer(self) -> None:
        """Reassigning land_mask through configure_context keeps the renderer in sync."""
        media_publisher = MediaPublisher()
        new_land_mask = LandMask(None)

        media_publisher.configure_context(land_mask=new_land_mask)

        assert media_publisher.land_mask is new_land_mask
        assert media_publisher._renderer.land_mask is new_land_mask

    def test_configure_context_ignores_unset_fields(self) -> None:
        """Omitted arguments leave existing plot_spec/land_mask state untouched."""
        media_publisher = MediaPublisher()
        media_publisher.configure_context(hemisphere="north")
        original_land_mask = media_publisher.land_mask

        media_publisher.configure_context()

        assert media_publisher.plot_spec.hemisphere == "north"
        assert media_publisher.land_mask is original_land_mask


class TestLogStaticInputs:
    def test_logs_images_for_each_input_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Render and log one image per variable, under the input_static prefix."""
        image = object()
        fake_render = MagicMock(return_value=image)
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_static", fake_render
        )
        image_logger = MagicMock()

        media_publisher = MediaPublisher()
        media_publisher.log_static_inputs(
            [fake_single_dataset()], TEST_DATES, [image_logger], prefix="validation"
        )

        assert fake_render.call_count == N_CHANNELS
        logged_keys = [c.kwargs["key"] for c in image_logger.log_image.call_args_list]
        assert logged_keys == [
            "validation/input_static/2020-01-01-example:ice_conc",
            "validation/input_static/2020-01-01-example:temperature",
        ]
        assert all(
            c.kwargs["images"] == [image] for c in image_logger.log_image.call_args_list
        )

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_static",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher()
        with caplog.at_level(logging.WARNING):
            media_publisher.log_static_inputs(
                [fake_single_dataset()], TEST_DATES, [MagicMock()]
            )

        assert "Static plotting skipped" in caplog.text

    def test_skips_on_generic_plotting_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow ValueError from the plotting layer and log a warning."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_static",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        media_publisher = MediaPublisher()
        with caplog.at_level(logging.WARNING):
            media_publisher.log_static_inputs(
                [fake_single_dataset()], TEST_DATES, [MagicMock()]
            )

        assert "Static plotting failed" in caplog.text


class TestLogStaticOutputs:
    def test_logs_images_per_channel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Render and log one image per output channel, keyed by date and variable name."""
        fake_render = MagicMock(return_value=object())
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_static", fake_render
        )
        image_logger = MagicMock()

        media_publisher = MediaPublisher()
        media_publisher.log_static_outputs(
            make_model_step_output(),
            TEST_DATES,
            [image_logger],
            channel_names=["sic"],
        )

        assert fake_render.call_count == N_CHANNELS
        assert image_logger.log_image.call_count == N_CHANNELS
        # Second channel has no configured name, so it falls back to channel_1.
        logged_keys = [c.kwargs["key"] for c in image_logger.log_image.call_args_list]
        assert logged_keys == [
            "output_static/2020-01-01-sic-difference",
            "output_static/2020-01-01-channel_1-difference",
        ]

    def test_includes_uncertainty_when_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Log an extra standardised-difference image for channels with uncertainty data."""
        fake_render = MagicMock(return_value=object())
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_static", fake_render
        )
        image_logger = MagicMock()
        uncertainties = {0: torch.zeros((N_TIMESTEPS, HEIGHT, WIDTH)).numpy()}

        media_publisher = MediaPublisher()
        media_publisher.log_static_outputs(
            make_model_step_output(),
            TEST_DATES,
            [image_logger],
            channel_names=["sic", "temperature"],
            uncertainties=uncertainties,
        )

        # Channel 0 (has uncertainty) renders twice: difference and z-score.
        # Channel 1 (no uncertainty) renders once: difference only.
        assert fake_render.call_count == N_CHANNELS + 1
        logged_keys = [c.kwargs["key"] for c in image_logger.log_image.call_args_list]
        assert logged_keys == [
            "output_static/2020-01-01-sic-difference",
            "output_static/2020-01-01-sic-z-score",
            "output_static/2020-01-01-temperature-difference",
        ]

        # The z-score render is the one with a norm set for its extra panel.
        z_score_call = fake_render.call_args_list[1]
        assert z_score_call.kwargs["norm"][-1] is not None
        for difference_call in (
            fake_render.call_args_list[0],
            fake_render.call_args_list[2],
        ):
            assert all(n is None for n in difference_call.kwargs["norm"])

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_static",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher()
        with caplog.at_level(logging.WARNING):
            media_publisher.log_static_outputs(
                make_model_step_output(),
                TEST_DATES,
                [MagicMock()],
                channel_names=["sic"],
            )

        assert "Static plotting skipped" in caplog.text

    def test_skips_on_generic_plotting_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow MemoryError from the plotting layer and log a warning."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_static",
            MagicMock(side_effect=MemoryError),
        )

        media_publisher = MediaPublisher()
        with caplog.at_level(logging.WARNING):
            media_publisher.log_static_outputs(
                make_model_step_output(),
                TEST_DATES,
                [MagicMock()],
                channel_names=["sic"],
            )

        assert "Static plotting failed" in caplog.text

    def test_preserves_prefix_and_channel_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Static routing keeps prefixes and fallback channel names stable."""
        image = object()
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_static",
            MagicMock(return_value=image),
        )
        image_logger = MagicMock()
        media_publisher = MediaPublisher(PlotSpec(selected_timestep=1))

        media_publisher.log_static_outputs(
            make_model_step_output(),
            TEST_DATES,
            [image_logger],
            channel_names=["ice_conc"],
            prefix="evaluate",
        )

        assert [c.kwargs for c in image_logger.log_image.call_args_list] == [
            {
                "key": "evaluate/output_static/2020-01-02-ice_conc-difference",
                "images": [image],
            },
            {
                "key": "evaluate/output_static/2020-01-02-channel_1-difference",
                "images": [image],
            },
        ]

    def test_without_prefix_uses_default_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Static routing keeps the established default logging namespace."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_static",
            MagicMock(return_value=object()),
        )
        image_logger = MagicMock()

        MediaPublisher(PlotSpec()).log_static_outputs(
            make_model_step_output(channels=1),
            TEST_DATES,
            [image_logger],
            channel_names=["ice_conc"],
        )

        assert (
            image_logger.log_image.call_args.kwargs["key"]
            == "output_static/2020-01-01-ice_conc-difference"
        )


class TestLogVideoInputs:
    def test_logs_videos_for_each_input_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plot and log one video per variable, under the input_video prefix."""
        buffer = MagicMock()
        fake_render = MagicMock(return_value=buffer)
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_video", fake_render
        )
        video_logger = MagicMock()

        media_publisher = MediaPublisher()
        media_publisher.log_video_inputs(
            [fake_single_dataset()], TEST_DATES, [video_logger], prefix="validation"
        )

        assert fake_render.call_count == N_CHANNELS
        assert video_logger.log_video.call_args_list == [
            call(
                key="validation/input_video/2020-01-01-example:ice_conc",
                videos=[buffer],
                format=[media_publisher.plot_spec.video_format],
            ),
            call(
                key="validation/input_video/2020-01-01-example:temperature",
                videos=[buffer],
                format=[media_publisher.plot_spec.video_format],
            ),
        ]

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_video",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher()
        with caplog.at_level(logging.WARNING):
            media_publisher.log_video_inputs(
                [fake_single_dataset()], TEST_DATES, [MagicMock()]
            )

        assert "Video plotting skipped" in caplog.text

    def test_skips_on_video_render_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow VideoRenderError and log a warning."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_video",
            MagicMock(side_effect=VideoRenderError("encoding failed")),
        )

        media_publisher = MediaPublisher()
        with caplog.at_level(logging.WARNING):
            media_publisher.log_video_inputs(
                [fake_single_dataset()], TEST_DATES, [MagicMock()]
            )

        assert "Video plotting skipped" in caplog.text

    def test_logs_exception_on_generic_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow a generic rendering error but log it at ERROR level with a traceback."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_video",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        media_publisher = MediaPublisher()
        with caplog.at_level(logging.ERROR):
            media_publisher.log_video_inputs(
                [fake_single_dataset()], TEST_DATES, [MagicMock()]
            )

        assert "Video plotting failed" in caplog.text
        assert caplog.records[-1].levelno == logging.ERROR


class TestLogVideoOutputs:
    def test_logs_videos_per_channel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Render and log one video per output channel, keyed by date and variable name."""
        fake_render = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_video", fake_render
        )
        video_logger = MagicMock()

        media_publisher = MediaPublisher()
        media_publisher.log_video_outputs(
            make_model_step_output(),
            TEST_DATES,
            [video_logger],
            channel_names=["sic"],
        )

        assert fake_render.call_count == N_CHANNELS
        assert video_logger.log_video.call_count == N_CHANNELS
        # Second channel has no configured name, so it falls back to channel_1.
        logged_keys = [c.kwargs["key"] for c in video_logger.log_video.call_args_list]
        assert logged_keys == [
            "output_video/2020-01-01-sic",
            "output_video/2020-01-01-channel_1",
        ]

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_video",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher()
        with caplog.at_level(logging.WARNING):
            media_publisher.log_video_outputs(
                make_model_step_output(),
                TEST_DATES,
                [MagicMock()],
                channel_names=["sic"],
            )

        assert "Video plotting skipped" in caplog.text

    def test_skips_on_video_render_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow VideoRenderError and log a warning."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_video",
            MagicMock(side_effect=VideoRenderError("encoding failed")),
        )

        media_publisher = MediaPublisher()
        with caplog.at_level(logging.WARNING):
            media_publisher.log_video_outputs(
                make_model_step_output(),
                TEST_DATES,
                [MagicMock()],
                channel_names=["sic"],
            )

        assert "Video plotting skipped" in caplog.text

    def test_logs_exception_on_generic_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow a generic rendering error but log it at ERROR level with a traceback."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_video",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        media_publisher = MediaPublisher()
        with caplog.at_level(logging.ERROR):
            media_publisher.log_video_outputs(
                make_model_step_output(),
                TEST_DATES,
                [MagicMock()],
                channel_names=["sic"],
            )

        assert "Video plotting failed" in caplog.text
        assert caplog.records[-1].levelno == logging.ERROR

    def test_rewinds_buffers_before_logging(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Video routing rewinds rendered buffers and forwards the format."""
        buffer = BytesIO(b"video-bytes")
        buffer.seek(5)

        monkeypatch.setattr(
            "icenet_mp.visualisations.panel_renderer.render_panels_video",
            lambda *args, **kwargs: buffer,  # noqa: ARG005
        )
        video_logger = MagicMock()
        media_publisher = MediaPublisher(PlotSpec(video_format="gif"))

        media_publisher.log_video_outputs(
            make_model_step_output(channels=1),
            TEST_DATES,
            [video_logger],
            channel_names=["ice_conc"],
            prefix="test",
        )

        video_logger.log_video.assert_called_once_with(
            key="test/output_video/2020-01-01-ice_conc",
            videos=[buffer],
            format=["gif"],
        )
        assert buffer.tell() == 0
