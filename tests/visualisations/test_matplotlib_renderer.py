"""Tests for the minimal panel-plot/panel-video rendering core."""

import logging
from io import BytesIO
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation
from matplotlib.figure import Figure
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import VideoRenderError
from icenet_mp.types import ArrayHW, ArrayTHW
from icenet_mp.visualisations.matplotlib_renderer import MatplotlibRenderer

renderer = MatplotlibRenderer()


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


class TestRenderPanels:
    def test_single_panel(self, era5_temperature_2d: ArrayHW) -> None:
        result = renderer.panels_static([era5_temperature_2d])

        assert isinstance(result, ImageFile)
        assert result.width > 0
        assert result.height > 0

    def test_three_panels_wider_than_one(
        self, era5_temperature_2d: ArrayHW, osisaf_ice_conc_2d: ArrayHW
    ) -> None:
        single = renderer.panels_static([era5_temperature_2d])
        triple = renderer.panels_static(
            [era5_temperature_2d, osisaf_ice_conc_2d, osisaf_ice_conc_2d]
        )

        assert triple.width > single.width

    def test_titles_applied(self, era5_temperature_2d: ArrayHW) -> None:
        result = renderer.panels_static(
            [era5_temperature_2d], panel_titles=["Ground Truth"]
        )

        assert isinstance(result, ImageFile)

    def test_shared_vmin_vmax(self, era5_temperature_2d: ArrayHW) -> None:
        result = renderer.panels_static([era5_temperature_2d], vmin=260.0, vmax=290.0)

        assert isinstance(result, ImageFile)

    def test_mismatched_titles_length_raises(
        self, era5_temperature_2d: ArrayHW
    ) -> None:
        with pytest.raises(ValueError, match=r"zip\(\)"):
            renderer.panels_static([era5_temperature_2d], panel_titles=["a", "b"])

    def test_per_panel_cmap_and_scale(
        self, era5_temperature_2d: ArrayHW, osisaf_ice_conc_2d: ArrayHW
    ) -> None:
        result = renderer.panels_static(
            [era5_temperature_2d, osisaf_ice_conc_2d],
            cmap=["RdBu_r", "Blues_r"],
            vmin=[260.0, 0.0],
            vmax=[290.0, 1.0],
        )

        assert isinstance(result, ImageFile)

    def test_suptitle(self, era5_temperature_2d: ArrayHW) -> None:
        result = renderer.panels_static(
            [era5_temperature_2d], figure_title="Shown: 2020-01-15"
        )

        assert isinstance(result, ImageFile)

    def test_contour_drawn_on_selected_panel(
        self, era5_temperature_2d: ArrayHW, osisaf_ice_conc_2d: ArrayHW
    ) -> None:
        """A contour_arrays entry draws a contour; a None entry draws none."""
        result = renderer.panels_static(
            [era5_temperature_2d, osisaf_ice_conc_2d],
            contour_arrays=[None, osisaf_ice_conc_2d],
            contour_level=0.15,
        )

        assert isinstance(result, ImageFile)

    def test_no_contour_without_level(self, osisaf_ice_conc_2d: ArrayHW) -> None:
        """contour_arrays without a contour_level draws nothing, not an error."""
        result = renderer.panels_static(
            [osisaf_ice_conc_2d], contour_arrays=[osisaf_ice_conc_2d]
        )

        assert isinstance(result, ImageFile)

    def test_contour_mismatched_length_raises(
        self, era5_temperature_2d: ArrayHW, osisaf_ice_conc_2d: ArrayHW
    ) -> None:
        with pytest.raises(ValueError, match=r"zip\(\)"):
            renderer.panels_static(
                [era5_temperature_2d, osisaf_ice_conc_2d],
                contour_arrays=[osisaf_ice_conc_2d],
                contour_level=0.15,
            )


class TestRenderPanelsVideo:
    def test_single_panel(self, era5_temperature_thw: ArrayTHW) -> None:
        result = renderer.panels_video([era5_temperature_thw])

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_three_panels(self, era5_temperature_thw: ArrayTHW) -> None:
        result = renderer.panels_video(
            [era5_temperature_thw, era5_temperature_thw, era5_temperature_thw]
        )

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_mismatched_titles_length_raises(
        self, era5_temperature_thw: ArrayTHW
    ) -> None:
        with pytest.raises(ValueError, match=r"zip\(\)"):
            renderer.panels_video([era5_temperature_thw], panel_titles=["a", "b"])

    def test_fps_is_configurable(self, era5_temperature_thw: ArrayTHW) -> None:
        result = renderer.panels_video([era5_temperature_thw], fps=4)

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_contour_redrawn_across_frames(
        self, era5_temperature_thw: ArrayTHW
    ) -> None:
        """A per-frame contour array renders without error across all frames."""
        result = renderer.panels_video(
            [era5_temperature_thw],
            contour_arrays=[era5_temperature_thw],
            contour_level=273.15,
        )

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0

    def test_contour_none_entry_skipped(self, era5_temperature_thw: ArrayTHW) -> None:
        """A None contour_arrays entry draws no contour on that panel."""
        result = renderer.panels_video(
            [era5_temperature_thw, era5_temperature_thw],
            contour_arrays=[None, era5_temperature_thw],
            contour_level=273.15,
        )

        assert isinstance(result, BytesIO)
        assert result.getbuffer().nbytes > 0


class TestImageFromFigure:
    def test_returns_image_with_positive_dimensions(self) -> None:
        """A real figure converts to a PIL image with a non-trivial size."""
        fig = make_figure()

        image = renderer._image_from_figure(fig, dpi=100)

        assert isinstance(image, ImageFile)
        assert image.width > 0
        assert image.height > 0


class TestVideoFromAnimation:
    def test_gif_output_has_gif_signature(self) -> None:
        """A GIF-format render returns a non-empty BytesIO starting with the GIF header."""
        fig = make_figure()
        anim = make_animation(fig)

        buffer = renderer._video_from_animation(
            anim, dpi=100, fps=2, video_format="gif"
        )

        assert isinstance(buffer, BytesIO)
        content = buffer.read()
        assert len(content) > 0
        assert content[:6] == b"GIF89a"

    def test_mp4_output_has_mp4_signature(self) -> None:
        """An MP4-format render returns a non-empty BytesIO containing an ftyp box."""
        fig = make_figure()
        anim = make_animation(fig)

        buffer = renderer._video_from_animation(
            anim, dpi=100, fps=2, video_format="mp4"
        )

        assert isinstance(buffer, BytesIO)
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
            renderer._video_from_animation(anim, dpi=100, fps=2, video_format="gif")

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
            renderer._video_from_animation(anim, dpi=100, fps=2, video_format="mp4")


class TestSuppressMplAnimationLogs:
    def test_sets_warning_level_and_restores_original(self) -> None:
        """The matplotlib.animation logger is raised to WARNING and restored after."""
        mpl_logger = logging.getLogger("matplotlib.animation")
        mpl_logger.setLevel(logging.INFO)

        with renderer._suppress_mpl_animation_logs():
            assert mpl_logger.level == logging.WARNING

        assert mpl_logger.level == logging.INFO
