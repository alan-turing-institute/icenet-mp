import logging
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, ClassVar, cast
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch

from icenet_mp.data import CombinedDataset, SingleDataset
from icenet_mp.exceptions import InvalidArrayError, VideoRenderError
from icenet_mp.types import Hemisphere, ModelStepOutput, PlotSpec
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.matplotlib_renderer import MatplotlibRenderer
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


def fake_metadata_input(name: str, variable_names: list[str]) -> SingleDataset:
    """Return a duck-typed SingleDataset stand-in exposing just name/variable_names.

    Used to populate `CombinedDataset.inputs` for `build_metadata`'s
    vars_by_source computation, distinct from `fake_single_dataset` above
    (which also implements array access for the rendering tests).
    """

    class FakeSingleDataset:
        def __init__(self, name: str, variable_names: list[str]) -> None:
            self.name = name
            self.variable_names = variable_names

    return cast("SingleDataset", FakeSingleDataset(name, variable_names))


def fake_combined_dataset(
    *,
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-10",
    frequency: np.timedelta64 | None = None,
    length: int = 10,
    n_history_steps: int = 0,
    inputs: list[SingleDataset] | None = None,
) -> CombinedDataset:
    """Return a duck-typed CombinedDataset stand-in exposing what build_metadata reads."""

    class FakeCombinedDataset:
        """Minimal CombinedDataset stand-in for MediaPublisher's metadata construction."""

        def __init__(self) -> None:
            self.start_date = np.datetime64(start_date)
            self.end_date = np.datetime64(end_date)
            self.frequency = (
                frequency if frequency is not None else np.timedelta64(1, "D")
            )
            self.n_history_steps = n_history_steps
            self.inputs = inputs if inputs is not None else []

        def __len__(self) -> int:
            return length

    return cast("CombinedDataset", FakeCombinedDataset())


def make_model_step_output(channels: int = N_CHANNELS) -> ModelStepOutput:
    """Build a ModelStepOutput with real tensors shaped [N, T, C, H, W]."""
    shape = (1, N_TIMESTEPS, channels, HEIGHT, WIDTH)
    return ModelStepOutput(
        prediction=torch.zeros(shape),
        target=torch.ones(shape),
        loss=torch.tensor(0.0),
    )


class TestBuildMetadata:
    @pytest.mark.parametrize(
        ("hours", "expected"),
        [
            (24, "daily"),
            (48, "2d"),
            (72, "3d"),
            (1, "hourly"),
            (6, "6h"),
            (0.5, "0.5h"),
        ],
    )
    def test_formats_hours_as_cadence_label(self, hours: float, expected: str) -> None:
        """Format a dataset's frequency into a short, human-readable cadence label."""
        frequency = np.timedelta64(int(hours * 60), "m")
        dataset = fake_combined_dataset(frequency=frequency)

        metadata = MediaPublisher.build_metadata(dataset)

        assert metadata.cadence == expected

    def test_derives_dates_cadence_and_length_from_dataset(self) -> None:
        """Metadata fields come from the dataset's realised state, not from config."""
        dataset = fake_combined_dataset(
            start_date="2020-01-01T12:30:00",
            end_date="2020-01-10T00:00:00",
            frequency=np.timedelta64(1, "D"),
            length=10,
            n_history_steps=3,
        )

        metadata = MediaPublisher.build_metadata(
            dataset, current_epoch=5, model_name="unet"
        )

        assert metadata.model == "unet"
        assert metadata.current_epoch == 5
        assert metadata.start == "2020-01-01"
        assert metadata.end == "2020-01-10"
        assert metadata.cadence == "daily"
        assert metadata.n_points == 10
        assert metadata.n_history_steps == 3

    def test_defaults_model_and_epoch_to_none(self) -> None:
        """Omitted model_name/current_epoch fall back to None."""
        dataset = fake_combined_dataset()

        metadata = MediaPublisher.build_metadata(dataset)

        assert metadata.model is None
        assert metadata.current_epoch is None

    def test_collects_sorted_variable_names_by_source(self) -> None:
        """vars_by_source maps each input dataset's name to its sorted variable names."""
        dataset = fake_combined_dataset(
            inputs=[
                fake_metadata_input("era5", ["sp", "2t"]),
                fake_metadata_input("osisaf-south", ["sic"]),
            ]
        )

        metadata = MediaPublisher.build_metadata(dataset)

        assert metadata.vars_by_source == {
            "era5": ["2t", "sp"],
            "osisaf-south": ["sic"],
        }

    def test_empty_inputs_yields_none_vars_by_source(self) -> None:
        """No input datasets means no variable-by-source mapping."""
        dataset = fake_combined_dataset(inputs=[])

        metadata = MediaPublisher.build_metadata(dataset)

        assert metadata.vars_by_source is None


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
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(video_format="mp4"),
        )
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
    def test_metadata_subtitle_reflects_constructor_dataset(self) -> None:
        """Metadata built from the constructor's dataset/epoch/model appears in the footer."""
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
            current_epoch=50,
            model_name="unet",
        )

        footer = media_publisher._panel_renderer._annotator.footer_for_static()

        assert "Model: unet" in footer
        assert "Epoch: 50" in footer

    def test_plot_spec_hemisphere_is_used_as_given(self) -> None:
        """Hemisphere is read straight from the given plot_spec, not set separately."""
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(hemisphere=Hemisphere.SOUTH),
        )

        assert media_publisher._plot_spec.hemisphere == "south"

    def test_land_mask_kwarg_is_used_by_the_renderer(self) -> None:
        """The given land_mask is passed straight through to the panel renderer."""
        new_land_mask = LandMask(None)

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            plot_spec=PlotSpec(),
            land_mask=new_land_mask,
        )

        assert media_publisher._panel_renderer.land_mask is new_land_mask

    def test_current_epoch_and_model_name_are_optional(self) -> None:
        """Omitting current_epoch/model_name leaves the footer without those lines."""
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )

        footer = media_publisher._panel_renderer._annotator.footer_for_static()

        assert "Model:" not in footer
        assert "Epoch:" not in footer


class TestLogStaticInputs:
    def test_logs_images_for_each_input_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Render and log one image per variable, under the input_static prefix."""
        image = object()
        fake_render = MagicMock(return_value=image)
        monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
        image_logger = MagicMock()

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            MatplotlibRenderer,
            "panels_static",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            MatplotlibRenderer,
            "panels_static",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        with caplog.at_level(logging.WARNING):
            media_publisher.log_static_inputs(
                [fake_single_dataset()], TEST_DATES, [MagicMock()]
            )

        assert "Static plotting failed" in caplog.text


class TestLogStaticOutputs:
    def test_logs_images_per_channel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Render and log one image per output channel, keyed by date and variable name."""
        fake_render = MagicMock(return_value=object())
        monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
        image_logger = MagicMock()

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            "output_static/2020-01-01-sic-truth-difference",
            "output_static/2020-01-01-channel_1-truth-difference",
        ]

    def test_includes_uncertainty_when_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Log an extra standardised-difference image for channels with uncertainty data."""
        fake_render = MagicMock(return_value=object())
        monkeypatch.setattr(MatplotlibRenderer, "panels_static", fake_render)
        image_logger = MagicMock()
        uncertainties = {0: torch.zeros((N_TIMESTEPS, HEIGHT, WIDTH)).numpy()}

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            "output_static/2020-01-01-sic-truth-difference",
            "output_static/2020-01-01-sic-z-score",
            "output_static/2020-01-01-temperature-truth-difference",
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
            MatplotlibRenderer,
            "panels_static",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            MatplotlibRenderer,
            "panels_static",
            MagicMock(side_effect=MemoryError),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            MatplotlibRenderer,
            "panels_static",
            MagicMock(return_value=image),
        )
        image_logger = MagicMock()
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(selected_timestep=1),
        )

        media_publisher.log_static_outputs(
            make_model_step_output(),
            TEST_DATES,
            [image_logger],
            channel_names=["ice_conc"],
            prefix="evaluate",
        )

        assert [c.kwargs for c in image_logger.log_image.call_args_list] == [
            {
                "key": "evaluate/output_static/2020-01-02-ice_conc-truth-difference",
                "images": [image],
            },
            {
                "key": "evaluate/output_static/2020-01-02-channel_1-truth-difference",
                "images": [image],
            },
        ]

    def test_without_prefix_uses_default_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Static routing keeps the established default logging namespace."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_static",
            MagicMock(return_value=object()),
        )
        image_logger = MagicMock()

        MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        ).log_static_outputs(
            make_model_step_output(channels=1),
            TEST_DATES,
            [image_logger],
            channel_names=["ice_conc"],
        )

        assert (
            image_logger.log_image.call_args.kwargs["key"]
            == "output_static/2020-01-01-ice_conc-truth-difference"
        )


class TestLogVideoInputs:
    def test_logs_videos_for_each_input_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plot and log one video per variable, under the input_video prefix."""
        buffer = MagicMock()
        fake_render = MagicMock(return_value=buffer)
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        video_logger = MagicMock()

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
        media_publisher.log_video_inputs(
            [fake_single_dataset()], TEST_DATES, [video_logger], prefix="validation"
        )

        assert fake_render.call_count == N_CHANNELS
        assert video_logger.log_video.call_args_list == [
            call(
                key="validation/input_video/2020-01-01-example:ice_conc",
                videos=[buffer],
                format=[media_publisher._plot_spec.video_format],
            ),
            call(
                key="validation/input_video/2020-01-01-example:temperature",
                videos=[buffer],
                format=[media_publisher._plot_spec.video_format],
            ),
        ]

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=VideoRenderError("encoding failed")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
        monkeypatch.setattr(MatplotlibRenderer, "panels_video", fake_render)
        video_logger = MagicMock()

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            "output_video/2020-01-01-sic-truth-difference",
            "output_video/2020-01-01-channel_1-truth-difference",
        ]

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=VideoRenderError("encoding failed")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            MatplotlibRenderer,
            "panels_video",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(),
        )
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
            MatplotlibRenderer,
            "panels_video",
            lambda *args, **kwargs: buffer,  # noqa: ARG005
        )
        video_logger = MagicMock()
        media_publisher = MediaPublisher(
            dataset=fake_combined_dataset(),
            land_mask=LandMask(None),
            plot_spec=PlotSpec(video_format="gif"),
        )

        media_publisher.log_video_outputs(
            make_model_step_output(channels=1),
            TEST_DATES,
            [video_logger],
            channel_names=["ice_conc"],
            prefix="test",
        )

        video_logger.log_video.assert_called_once_with(
            key="test/output_video/2020-01-01-ice_conc-truth-difference",
            videos=[buffer],
            format=["gif"],
        )
        assert buffer.tell() == 0
