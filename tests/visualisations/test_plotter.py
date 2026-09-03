import logging
from datetime import datetime
from typing import ClassVar, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.data import SingleDataset
from icenet_mp.exceptions import InvalidArrayError, VideoRenderError
from icenet_mp.types import Metadata, ModelStepOutput, PlotSpec
from icenet_mp.visualisations.plotter import Plotter

N_TIMESTEPS = 2
N_CHANNELS = 2
HEIGHT = 4
WIDTH = 4

TEST_DATES = [datetime(2020, 1, 1), datetime(2020, 1, 2)]


class FakeInputDataset:
    """Minimal SingleDataset stand-in exposing the attributes Plotter reads."""

    name = "example"
    variable_names: ClassVar[list[str]] = ["ice_conc", "temperature"]

    def __getitem__(self, _idx: int) -> np.ndarray:
        """Return a deterministic [C, H, W] array for the requested timestep."""
        return np.ones((N_CHANNELS, HEIGHT, WIDTH), dtype=np.float32)

    def get_tchw(self, dates: list) -> np.ndarray:
        """Return a deterministic [T, C, H, W] array for the requested dates."""
        return np.ones((len(dates), N_CHANNELS, HEIGHT, WIDTH), dtype=np.float32)


def fake_input_dataset() -> SingleDataset:
    """Return a duck-typed SingleDataset stand-in, cast to satisfy Plotter's typing."""
    return cast("SingleDataset", FakeInputDataset())


def make_model_step_output() -> ModelStepOutput:
    """Build a ModelStepOutput with real tensors shaped [N, T, C, H, W]."""
    shape = (1, N_TIMESTEPS, N_CHANNELS, HEIGHT, WIDTH)
    return ModelStepOutput(
        prediction=torch.zeros(shape),
        target=torch.ones(shape),
        loss=torch.tensor(0.0),
    )


@pytest.fixture
def plotter() -> Plotter:
    """A Plotter with a default PlotSpec."""
    return Plotter(PlotSpec())


class TestMetadataAndHemisphere:
    def test_get_metadata_delegates_to_build_metadata(
        self, plotter: Plotter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forward the config and model name to build_metadata."""
        expected = Metadata(model="unet")
        fake_build_metadata = MagicMock(return_value=expected)
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.build_metadata", fake_build_metadata
        )
        config = DictConfig({})

        result = plotter.get_metadata(config, "unet")

        fake_build_metadata.assert_called_once_with(config, "unet")
        assert result is expected

    def test_set_metadata_updates_plot_spec_subtitle(
        self, plotter: Plotter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Format the metadata and store it as the plot spec subtitle."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.format_metadata_subtitle",
            MagicMock(return_value="epochs=50"),
        )

        plotter.set_metadata(Metadata(model="unet"))

        assert plotter.plot_spec.metadata_subtitle == "epochs=50"

    def test_set_hemisphere_updates_plot_spec(self, plotter: Plotter) -> None:
        """Store the hemisphere on the plot spec."""
        plotter.set_hemisphere("south")

        assert plotter.plot_spec.hemisphere == "south"


class TestLogStaticInputs:
    def test_logs_images_for_each_input_dataset(
        self, plotter: Plotter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plot and log one image group per variable, under the input_static prefix."""
        fake_plot = MagicMock(return_value={"ice_conc": [object()]})
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_inputs", fake_plot
        )
        image_logger = MagicMock()

        plotter.log_static_inputs(
            [fake_input_dataset()], TEST_DATES, [image_logger], prefix="validation"
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
        plotter: Plotter,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_inputs",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        with caplog.at_level(logging.WARNING):
            plotter.log_static_inputs([fake_input_dataset()], TEST_DATES, [MagicMock()])

        assert "Static plotting skipped" in caplog.text

    def test_skips_on_generic_plotting_error(
        self,
        plotter: Plotter,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Swallow ValueError from the plotting layer and log a warning."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_inputs",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        with caplog.at_level(logging.WARNING):
            plotter.log_static_inputs([fake_input_dataset()], TEST_DATES, [MagicMock()])

        assert "Static plotting failed" in caplog.text


class TestLogStaticOutputs:
    def test_logs_images_per_channel(
        self, plotter: Plotter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plot and log one image group per output channel, named from channel_names."""
        fake_plot = MagicMock(return_value={"comparison": [object()]})
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_prediction", fake_plot
        )
        image_logger = MagicMock()

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
        self, plotter: Plotter, monkeypatch: pytest.MonkeyPatch
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
        plotter: Plotter,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_prediction",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

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
        plotter: Plotter,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Swallow MemoryError from the plotting layer and log a warning."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_static_prediction",
            MagicMock(side_effect=MemoryError),
        )

        with caplog.at_level(logging.WARNING):
            plotter.log_static_outputs(
                make_model_step_output(),
                TEST_DATES,
                [MagicMock()],
                channel_names=["sic"],
            )

        assert "Static plotting failed" in caplog.text


class TestLogVideoInputs:
    def test_logs_videos_for_each_input_dataset(
        self, plotter: Plotter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plot and log one video group per variable, under the input_video prefix."""
        buffer = MagicMock()
        fake_plot = MagicMock(return_value={"ice_conc": buffer})
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_inputs", fake_plot
        )
        video_logger = MagicMock()

        plotter.log_video_inputs(
            [fake_input_dataset()], TEST_DATES, [video_logger], prefix="validation"
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
        plotter: Plotter,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_inputs",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

        with caplog.at_level(logging.WARNING):
            plotter.log_video_inputs([fake_input_dataset()], TEST_DATES, [MagicMock()])

        assert "Video plotting skipped" in caplog.text

    def test_skips_on_video_render_error(
        self,
        plotter: Plotter,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Swallow VideoRenderError and log a warning."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_inputs",
            MagicMock(side_effect=VideoRenderError("encoding failed")),
        )

        with caplog.at_level(logging.WARNING):
            plotter.log_video_inputs([fake_input_dataset()], TEST_DATES, [MagicMock()])

        assert "Video plotting skipped" in caplog.text

    def test_logs_exception_on_generic_error(
        self,
        plotter: Plotter,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Swallow a generic rendering error but log it at ERROR level with a traceback."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_inputs",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        with caplog.at_level(logging.ERROR):
            plotter.log_video_inputs([fake_input_dataset()], TEST_DATES, [MagicMock()])

        assert "Video plotting failed" in caplog.text
        assert caplog.records[-1].levelno == logging.ERROR


class TestLogVideoOutputs:
    def test_logs_videos_per_channel(
        self, plotter: Plotter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plot and log one video group per output channel, named from channel_names."""
        buffer = MagicMock()
        fake_plot = MagicMock(return_value={"forecast": buffer})
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_prediction", fake_plot
        )
        video_logger = MagicMock()

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
        plotter: Plotter,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Swallow InvalidArrayError and log a warning instead of raising."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_prediction",
            MagicMock(side_effect=InvalidArrayError("bad array")),
        )

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
        plotter: Plotter,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Swallow VideoRenderError and log a warning."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_prediction",
            MagicMock(side_effect=VideoRenderError("encoding failed")),
        )

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
        plotter: Plotter,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Swallow a generic rendering error but log it at ERROR level with a traceback."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plotter.plot_video_prediction",
            MagicMock(side_effect=ValueError("bad shape")),
        )

        with caplog.at_level(logging.ERROR):
            plotter.log_video_outputs(
                make_model_step_output(),
                TEST_DATES,
                [MagicMock()],
                channel_names=["sic"],
            )

        assert "Video plotting failed" in caplog.text
        assert caplog.records[-1].levelno == logging.ERROR
