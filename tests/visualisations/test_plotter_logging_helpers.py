from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from icenet_mp.visualisations.plotter import Plotter


def test_log_path_handles_optional_prefix() -> None:
    """Build the same namespaces for prefixed and unprefixed logging."""
    assert Plotter._log_path(None, "output_static") == "output_static"
    assert Plotter._log_path("test", "output_static") == "test/output_static"


def test_channel_name_uses_stable_fallback() -> None:
    """Use configured names when available and indexed fallbacks otherwise."""
    assert Plotter._channel_name(["sic"], 0) == "sic"
    assert Plotter._channel_name(["sic"], 2) == "channel_2"


def test_log_images_fans_out_to_all_loggers() -> None:
    """Send every image group to each configured logger."""
    first = MagicMock()
    second = MagicMock()
    images = {"comparison": [object()], "error": [object()]}

    Plotter._log_images(images, [first, second], "validation/output_static")

    expected = [
        call(key="validation/output_static/comparison", images=images["comparison"]),
        call(key="validation/output_static/error", images=images["error"]),
    ]
    assert first.log_image.call_args_list == expected
    assert second.log_image.call_args_list == expected


def test_log_videos_rewinds_for_each_logger_and_preserves_format() -> None:
    """Rewind shared buffers before every logger handoff."""
    plotter = Plotter(SimpleNamespace(video_format="mp4"))  # type: ignore[arg-type]
    first = MagicMock()
    second = MagicMock()
    buffer = BytesIO(b"video")
    buffer.seek(3)

    plotter._log_videos(
        {"forecast": buffer},
        [first, second],
        "test/output_video",
    )

    expected = call(
        key="test/output_video/forecast",
        videos=[buffer],
        format=["mp4"],
    )
    first.log_video.assert_called_once_with(*expected.args, **expected.kwargs)
    second.log_video.assert_called_once_with(*expected.args, **expected.kwargs)
    assert buffer.tell() == 0
