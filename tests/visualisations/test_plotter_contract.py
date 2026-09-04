from datetime import UTC, datetime
from io import BytesIO

import pytest
import torch

import icenet_mp.visualisations.plotter as plotter_module
from icenet_mp.types import ModelStepOutput, PlotSpec
from icenet_mp.visualisations import Plotter


class ImageLogger:
    def __init__(self) -> None:
        """Initialise recorded image-log calls."""
        self.calls: list[tuple[str, list[object]]] = []

    def log_image(self, *, key: str, images: list[object]) -> None:
        self.calls.append((key, images))


class VideoLogger:
    def __init__(self) -> None:
        """Initialise recorded video-log calls."""
        self.calls: list[tuple[str, int, list[str]]] = []

    def log_video(
        self, *, key: str, videos: list[BytesIO], **kwargs: list[str]
    ) -> None:
        self.calls.append((key, videos[0].tell(), kwargs["format"]))


def _outputs(channels: int = 2) -> ModelStepOutput:
    shape = (1, 2, channels, 2, 2)
    return ModelStepOutput(
        prediction=torch.zeros(shape),
        target=torch.ones(shape),
        loss=torch.tensor(0.0),
    )


def _dates() -> list[datetime]:
    return [
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
    ]


def test_static_output_routing_preserves_prefix_and_channel_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static routing keeps prefixes and fallback channel names stable."""
    seen_variables: list[str] = []
    image = object()

    def fake_plot_static_prediction(*args, variable_name: str, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
        seen_variables.append(variable_name)
        return {"forecast": [image]}

    monkeypatch.setattr(
        plotter_module, "plot_static_prediction", fake_plot_static_prediction
    )
    logger = ImageLogger()
    plotter = Plotter(PlotSpec(selected_timestep=1))

    plotter.log_static_outputs(
        _outputs(),
        _dates(),
        [logger],
        channel_names=["ice_conc"],
        prefix="evaluate",
    )

    assert seen_variables == ["ice_conc", "channel_1"]
    assert logger.calls == [
        ("evaluate/output_static/forecast", [image]),
        ("evaluate/output_static/forecast", [image]),
    ]


def test_static_output_routing_without_prefix_uses_default_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static routing keeps the established default logging namespace."""
    monkeypatch.setattr(
        plotter_module,
        "plot_static_prediction",
        lambda *args, **kwargs: {"map": [object()]},  # noqa: ARG005
    )
    logger = ImageLogger()

    Plotter(PlotSpec()).log_static_outputs(
        _outputs(channels=1),
        _dates(),
        [logger],
        channel_names=["ice_conc"],
    )

    assert logger.calls[0][0] == "output_static/map"


def test_video_output_routing_rewinds_buffers_before_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Video routing rewinds rendered buffers and forwards the format."""
    buffer = BytesIO(b"video-bytes")
    buffer.seek(5)

    monkeypatch.setattr(
        plotter_module,
        "plot_video_prediction",
        lambda *args, **kwargs: {"forecast": buffer},  # noqa: ARG005
    )
    logger = VideoLogger()
    plotter = Plotter(PlotSpec(video_format="gif"))

    plotter.log_video_outputs(
        _outputs(channels=1),
        _dates(),
        [logger],
        channel_names=["ice_conc"],
        prefix="test",
    )

    assert logger.calls == [("test/output_video/forecast", 0, ["gif"])]


def test_set_hemisphere_updates_plot_spec() -> None:
    """Plotter keeps hemisphere state on its PlotSpec."""
    plotter = Plotter(PlotSpec())

    plotter.set_hemisphere("south")

    assert plotter.plot_spec.hemisphere == "south"
