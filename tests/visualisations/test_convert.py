import io
import logging
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation
from matplotlib.figure import Figure
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import VideoRenderError
from icenet_mp.visualisations.convert import (
    _suppress_mpl_animation_logs,
    image_from_figure,
    video_from_animation,
)


def make_figure() -> Figure:
    """Build a small real matplotlib figure for conversion tests."""
    fig, ax = plt.subplots()
    ax.imshow(np.random.default_rng(0).random((8, 8)))
    return fig


def make_animation(fig: Figure) -> animation.FuncAnimation:
    """Build a minimal real FuncAnimation with a trivial per-frame update."""
    ax = fig.axes[0]
    image = ax.imshow(np.zeros((8, 8)))

    def animate(frame: int) -> tuple[()]:
        image.set_data(np.full((8, 8), frame))
        return ()

    return animation.FuncAnimation(fig, animate, frames=2, interval=200, blit=False)


class TestImageFromFigure:
    def test_returns_image_with_positive_dimensions(self) -> None:
        """A real figure converts to a PIL image with a non-trivial size."""
        fig = make_figure()

        image = image_from_figure(fig, dpi=100)

        assert isinstance(image, ImageFile)
        assert image.width > 0
        assert image.height > 0


class TestVideoFromAnimation:
    def test_gif_output_has_gif_signature(self) -> None:
        """A GIF-format render returns a non-empty BytesIO starting with the GIF header."""
        fig = make_figure()
        anim = make_animation(fig)

        buffer = video_from_animation(anim, dpi=100, fps=2, video_format="gif")

        assert isinstance(buffer, io.BytesIO)
        content = buffer.read()
        assert len(content) > 0
        assert content[:6] == b"GIF89a"

    def test_mp4_output_has_mp4_signature(self) -> None:
        """An MP4-format render returns a non-empty BytesIO containing an ftyp box."""
        fig = make_figure()
        anim = make_animation(fig)

        buffer = video_from_animation(anim, dpi=100, fps=2, video_format="mp4")

        assert isinstance(buffer, io.BytesIO)
        content = buffer.read()
        assert len(content) > 0
        assert b"ftyp" in content[:100]

    def test_save_failure_raises_video_render_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An OSError from anim.save is wrapped and re-raised as VideoRenderError."""
        fig = make_figure()
        anim = make_animation(fig)
        monkeypatch.setattr(anim, "save", MagicMock(side_effect=OSError("disk full")))

        with pytest.raises(VideoRenderError, match="Video encoding failed"):
            video_from_animation(anim, dpi=100, fps=2, video_format="gif")

    def test_save_memory_error_raises_video_render_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A MemoryError from anim.save is wrapped and re-raised as VideoRenderError."""
        fig = make_figure()
        anim = make_animation(fig)
        monkeypatch.setattr(
            anim, "save", MagicMock(side_effect=MemoryError("out of memory"))
        )

        with pytest.raises(VideoRenderError, match="Video encoding failed"):
            video_from_animation(anim, dpi=100, fps=2, video_format="mp4")


class TestSuppressMplAnimationLogs:
    def test_sets_warning_level_and_restores_original(self) -> None:
        """The matplotlib.animation logger is raised to WARNING and restored after."""
        mpl_logger = logging.getLogger("matplotlib.animation")
        mpl_logger.setLevel(logging.INFO)

        with _suppress_mpl_animation_logs():
            assert mpl_logger.level == logging.WARNING

        assert mpl_logger.level == logging.INFO
