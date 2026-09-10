import logging
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, ClassVar, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.data import SingleDataset
from icenet_mp.exceptions import InvalidArrayError, VideoRenderError
from icenet_mp.types import Metadata, ModelStepOutput, PlotSpec
from icenet_mp.visualisations.plotter import Plotter

if TYPE_CHECKING:
    from PIL.ImageFile import ImageFile

N_TIMESTEPS = 2
N_CHANNELS = 2
HEIGHT = 4
WIDTH = 4

TEST_DATES = [datetime(2020, 1, 1), datetime(2020, 1, 2)]


def fake_single_dataset() -> SingleDataset:
    """Return a duck-typed SingleDataset stand-in, cast to satisfy Plotter's typing."""

    class FakeSingleDataset:
        """Minimal SingleDataset stand-in exposing the attributes Plotter reads."""

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
        assert Plotter._log_path(None, "output_static") == "output_static"
        assert Plotter._log_path("test", "output_static") == "test/output_static"

    def test_channel_name_uses_stable_fallback(self) -> None:
        """Use configured names when available and indexed fallbacks otherwise."""
        assert Plotter._channel_name(["sic"], 0) == "sic"
        assert Plotter._channel_name(["sic"], 2) == "channel_2"

    def test_log_images_fans_out_to_all_loggers(self) -> None:
        """Send every image group to each configured logger."""
        first = MagicMock()
        second = MagicMock()
        images = cast(
            "dict[str, list[ImageFile]]",
            {"comparison": [object()], "error": [object()]},
        )

        Plotter._log_images(images, [first, second], "validation/output_static")

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
        plotter = Plotter(PlotSpec(video_format="mp4"))
        first = MagicMock()
        second = MagicMock()
        buffer = BytesIO(b"video")
        buffer.seek(3)

        plotter._log_videos(
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
    def test_get_metadata_delegates_to_build_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forward the config and model name to build_metadata."""
        expected = Metadata(model="unet")
        fake_build_metadata = MagicMock(return_value=expected)
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.build_metadata", fake_build_metadata
        )
        config = DictConfig({})

        plotter = Plotter()
        result = plotter.get_metadata(config, "unet")

        fake_build_metadata.assert_called_once_with(config, "unet")
        assert result is expected

    def test_set_metadata_updates_plot_spec_subtitle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Format the metadata and store it as the plot spec subtitle."""
        plotter = Plotter()
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.format_metadata_subtitle",
            MagicMock(return_value="epochs=50"),
        )

        plotter.set_metadata(Metadata(model="unet"))

        assert plotter.plot_spec.metadata_subtitle == "epochs=50"

    def test_set_hemisphere_updates_plot_spec(self) -> None:
        """Plotter keeps hemisphere state on its PlotSpec."""
        plotter = Plotter()
        plotter.set_hemisphere("south")

        assert plotter.plot_spec.hemisphere == "south"


class TestLogStaticInputs:
    def test_logs_images_for_each_input_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plot and log one image group per variable, under the input_static prefix."""
        fake_plot = MagicMock(return_value={"ice_conc": [object()]})
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_inputs", fake_plot
        )
        image_logger = MagicMock()

        plotter = Plotter()
        plotter.log_static_inputs(
            [fake_single_dataset()], TEST_DATES, [image_logger], prefix="validation"
        )

        fake_plot.assert_called_once()
        assert fake_plot.call_args.kwargs["when"] == TEST_DATES[0]
        variables = fake_plot.call_args.args[0]
        assert set(variables) == {"example:ice_conc", "example:temperature"}
        image_logger.log_image.assert_called_once_with(
            key="validation/input_static/ice_conc",
            images=[fake_plot.return_value["ice_conc"][0]],
        )

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_inputs",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        plotter = Plotter()
        with caplog.at_level(logging.WARNING):
            plotter.log_static_inputs(
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
            "icenet_mp.visualisations.plotter.plot_static_inputs",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        plotter = Plotter()
        with caplog.at_level(logging.WARNING):
            plotter.log_static_inputs(
                [fake_single_dataset()], TEST_DATES, [MagicMock()]
            )

        assert "Static plotting failed" in caplog.text


class TestLogStaticOutputs:
    def test_logs_images_per_channel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plot and log one image group per output channel, named from channel_names."""
        fake_plot = MagicMock(return_value={"comparison": [object()]})
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_prediction", fake_plot
        )
        image_logger = MagicMock()

        plotter = Plotter()
        plotter.log_static_outputs(
            make_model_step_output(),
            TEST_DATES,
            [image_logger],
            channel_names=["sic"],
        )

        assert fake_plot.call_count == N_CHANNELS
        # Second channel has no configured name, so it falls back to channel_1.
        assert fake_plot.call_args_list[0].kwargs["variable_name"] == "sic"
        assert fake_plot.call_args_list[1].kwargs["variable_name"] == "channel_1"
        assert image_logger.log_image.call_count == N_CHANNELS

    def test_includes_uncertainty_when_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Merge uncertainty images into the logged output for channels with data."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_prediction",
            MagicMock(side_effect=lambda *_a, **_kw: {"comparison": [object()]}),
        )
        fake_uncertainty = MagicMock(return_value={"uncertainty": [object()]})
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_uncertainty", fake_uncertainty
        )
        image_logger = MagicMock()
        uncertainties = {0: torch.zeros((N_TIMESTEPS, HEIGHT, WIDTH)).numpy()}

        plotter = Plotter()
        plotter.log_static_outputs(
            make_model_step_output(),
            TEST_DATES,
            [image_logger],
            channel_names=["sic", "temperature"],
            uncertainties=uncertainties,
        )

        # Only channel 0 has an uncertainty array.
        fake_uncertainty.assert_called_once()
        assert fake_uncertainty.call_args.kwargs["variable_name"] == "sic"
        logged_keys = [
            call.kwargs["key"] for call in image_logger.log_image.call_args_list
        ]
        assert "output_static/uncertainty" in logged_keys
        assert logged_keys.count("output_static/uncertainty") == 1

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_prediction",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        plotter = Plotter()
        with caplog.at_level(logging.WARNING):
            plotter.log_static_outputs(
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
            "icenet_mp.visualisations.plotter.plot_static_prediction",
            MagicMock(side_effect=MemoryError),
        )

        plotter = Plotter()
        with caplog.at_level(logging.WARNING):
            plotter.log_static_outputs(
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
        seen_variables: list[str] = []
        image = object()

        def fake_plot_static_prediction(*args, variable_name: str, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
            seen_variables.append(variable_name)
            return {"forecast": [image]}

        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_prediction",
            fake_plot_static_prediction,
        )
        image_logger = MagicMock()
        plotter = Plotter(PlotSpec(selected_timestep=1))

        plotter.log_static_outputs(
            make_model_step_output(),
            TEST_DATES,
            [image_logger],
            channel_names=["ice_conc"],
            prefix="evaluate",
        )

        assert seen_variables == ["ice_conc", "channel_1"]
        assert [c.kwargs for c in image_logger.log_image.call_args_list] == [
            {"key": "evaluate/output_static/forecast", "images": [image]},
            {"key": "evaluate/output_static/forecast", "images": [image]},
        ]

    def test_without_prefix_uses_default_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Static routing keeps the established default logging namespace."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_prediction",
            lambda *args, **kwargs: {"map": [object()]},  # noqa: ARG005
        )
        image_logger = MagicMock()

        Plotter(PlotSpec()).log_static_outputs(
            make_model_step_output(channels=1),
            TEST_DATES,
            [image_logger],
            channel_names=["ice_conc"],
        )

        assert image_logger.log_image.call_args.kwargs["key"] == "output_static/map"


class TestLogVideoInputs:
    def test_logs_videos_for_each_input_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plot and log one video group per variable, under the input_video prefix."""
        buffer = MagicMock()
        fake_plot = MagicMock(return_value={"ice_conc": buffer})
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_inputs", fake_plot
        )
        video_logger = MagicMock()

        plotter = Plotter()
        plotter.log_video_inputs(
            [fake_single_dataset()], TEST_DATES, [video_logger], prefix="validation"
        )

        variables = fake_plot.call_args.args[0]
        assert set(variables) == {"example:ice_conc", "example:temperature"}
        video_logger.log_video.assert_called_once_with(
            key="validation/input_video/ice_conc",
            videos=[buffer],
            format=[plotter.plot_spec.video_format],
        )

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_inputs",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        plotter = Plotter()
        with caplog.at_level(logging.WARNING):
            plotter.log_video_inputs([fake_single_dataset()], TEST_DATES, [MagicMock()])

        assert "Video plotting skipped" in caplog.text

    def test_skips_on_video_render_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow VideoRenderError and log a warning."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_inputs",
            MagicMock(side_effect=VideoRenderError("encoding failed")),
        )

        plotter = Plotter()
        with caplog.at_level(logging.WARNING):
            plotter.log_video_inputs([fake_single_dataset()], TEST_DATES, [MagicMock()])

        assert "Video plotting skipped" in caplog.text

    def test_logs_exception_on_generic_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow a generic rendering error but log it at ERROR level with a traceback."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_inputs",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        plotter = Plotter()
        with caplog.at_level(logging.ERROR):
            plotter.log_video_inputs([fake_single_dataset()], TEST_DATES, [MagicMock()])

        assert "Video plotting failed" in caplog.text
        assert caplog.records[-1].levelno == logging.ERROR


class TestLogVideoOutputs:
    def test_logs_videos_per_channel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plot and log one video group per output channel, named from channel_names."""
        buffer = MagicMock()
        fake_plot = MagicMock(return_value={"forecast": buffer})
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_prediction", fake_plot
        )
        video_logger = MagicMock()

        plotter = Plotter()
        plotter.log_video_outputs(
            make_model_step_output(),
            TEST_DATES,
            [video_logger],
            channel_names=["sic"],
        )

        assert fake_plot.call_count == N_CHANNELS
        assert fake_plot.call_args_list[0].kwargs["variable_name"] == "sic"
        assert fake_plot.call_args_list[1].kwargs["variable_name"] == "channel_1"
        assert video_logger.log_video.call_count == N_CHANNELS

    def test_skips_on_invalid_array_error(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_prediction",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        plotter = Plotter()
        with caplog.at_level(logging.WARNING):
            plotter.log_video_outputs(
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
            "icenet_mp.visualisations.plotter.plot_video_prediction",
            MagicMock(side_effect=VideoRenderError("encoding failed")),
        )

        plotter = Plotter()
        with caplog.at_level(logging.WARNING):
            plotter.log_video_outputs(
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
            "icenet_mp.visualisations.plotter.plot_video_prediction",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        plotter = Plotter()
        with caplog.at_level(logging.ERROR):
            plotter.log_video_outputs(
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
            "icenet_mp.visualisations.plotter.plot_video_prediction",
            lambda *args, **kwargs: {"forecast": buffer},  # noqa: ARG005
        )
        video_logger = MagicMock()
        plotter = Plotter(PlotSpec(video_format="gif"))

        plotter.log_video_outputs(
            make_model_step_output(channels=1),
            TEST_DATES,
            [video_logger],
            channel_names=["ice_conc"],
            prefix="test",
        )

        video_logger.log_video.assert_called_once_with(
            key="test/output_video/forecast", videos=[buffer], format=["gif"]
        )
        assert buffer.tell() == 0
