"""Minimal panel-plot and panel-video rendering core."""

from collections.abc import Sequence
from io import BytesIO
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from PIL.ImageFile import ImageFile

from icenet_mp.types import ArrayHW, ArrayTHW
from icenet_mp.utils import to_list

from .convert import image_from_figure, video_from_animation

_COLOURBAR_ASPECT = 25
_COLOURBAR_LABEL_SIZE = 9
_PANEL_HEIGHT_IN = 6


def _add_colourbar(fig: Figure, image: AxesImage, axes: Axes | list[Axes]) -> None:
    """Add a horizontal colourbar sized/labelled consistently regardless of span.

    `aspect` (long:short-axis ratio) is scaled by how many panels the colourbar
    spans, so a colourbar shared across N panels comes out the same thickness
    as one attached to a single panel, not N times thicker.
    """
    span = len(axes) if isinstance(axes, list) else 1
    cbar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        aspect=_COLOURBAR_ASPECT * span,
        pad=0.04,
    )
    cbar.ax.tick_params(labelsize=_COLOURBAR_LABEL_SIZE)


def render_panels(  # noqa: PLR0913
    arrays: Sequence[ArrayHW],
    *,
    cmap: str | Sequence[str] = "viridis",
    figure_title: str | None = None,
    footer_text: str | None = None,
    group_axes: tuple[int, int] | None = None,
    panel_titles: Sequence[str] | None = None,
    vmax: float | Sequence[float | None] | None = None,
    vmin: float | Sequence[float | None] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Render multiple panels side-by-side.

    Args:
        arrays: 1 to 3 2D `[H, W]` arrays, one per panel.
        cmap: Optional colourmap(s), either shared by all panels or one per panel.
        figure_title: Optional figure-level title drawn above all panels.
        footer_text: Optional footer text drawn below the colourbars.
        group_axes: Optional inclusive `(start, end)` panel index range to share a colourbar.
        panel_titles: Optional per-panel title, one per panel.
        vmax: Optional upper colour-scale bound(s), either shared or one per panel.
        vmin: Optional lower colour-scale bound(s), either shared or one per panel.

    Returns:
        The matplotlib Figure, and the list of panel Axes in panel order (each
        Axes' drawn image is available as `ax.images[0]`, e.g. for animation).

    """
    n = len(arrays)
    cmaps = (to_list(cmap) * n)[:n]
    vmins = list(vmin) if isinstance(vmin, Sequence) else [vmin] * n
    vmaxs = list(vmax) if isinstance(vmax, Sequence) else [vmax] * n

    fig, axes_ = plt.subplots(
        1, n, figsize=(_PANEL_HEIGHT_IN * n, _PANEL_HEIGHT_IN), layout="compressed"
    )
    axes: list[Axes] = np.atleast_1d(axes_).tolist()
    for ax, array in zip(axes, arrays, strict=True):
        ax.set_box_aspect(array.shape[0] / array.shape[1])

    images = [
        ax.imshow(arr, cmap=c, vmin=lo, vmax=hi, origin="upper")
        for ax, arr, c, lo, hi in zip(axes, arrays, cmaps, vmins, vmaxs, strict=True)
    ]
    for ax, title in zip(axes, panel_titles or [""] * n, strict=True):
        ax.set_title(title)
        ax.axis("off")

    if group_axes is not None:
        start, end = group_axes
        _add_colourbar(fig, images[start], axes[start : end + 1])
        for i, (ax, image) in enumerate(zip(axes, images, strict=True)):
            if i < start or i > end:
                _add_colourbar(fig, image, ax)
    else:
        for ax, image in zip(axes, images, strict=True):
            _add_colourbar(fig, image, ax)

    if figure_title:
        fig.suptitle(figure_title)
    if footer_text:
        fig.supxlabel(footer_text)

    return fig, axes


def render_panels_static(  # noqa: PLR0913
    arrays: Sequence[ArrayHW],
    *,
    cmap: str | Sequence[str] = "viridis",
    dpi: int = 150,
    figure_title: str | None = None,
    footer_text: str | None = None,
    group_axes: tuple[int, int] | None = None,
    panel_titles: Sequence[str] | None = None,
    vmax: float | Sequence[float | None] | None = None,
    vmin: float | Sequence[float | None] | None = None,
) -> ImageFile:
    """Render multiple panels side-by-side.

    Args:
        arrays: 1 to 3 2D `[H, W]` arrays, one per panel.
        cmap: Optional colourmap(s), either shared by all panels or one per panel.
        dpi: Dots per inch for the rendered image.
        figure_title: Optional figure-level title drawn above all panels.
        footer_text: Optional footer text drawn below the colourbars.
        group_axes: Optional inclusive `(start, end)` panel index range to share a colourbar.
        panel_titles: Optional per-panel title, one per panel.
        vmax: Optional upper colour-scale bound(s), either shared or one per panel.
        vmin: Optional lower colour-scale bound(s), either shared or one per panel.

    Returns:
        A PIL image of the rendered panels.

    """
    figure, _ = render_panels(
        arrays,
        cmap=cmap,
        figure_title=figure_title,
        footer_text=footer_text,
        group_axes=group_axes,
        panel_titles=panel_titles,
        vmax=vmax,
        vmin=vmin,
    )
    try:
        return image_from_figure(figure, dpi=dpi)
    finally:
        plt.close(figure)


def render_panels_video(  # noqa: PLR0913
    arrays: Sequence[ArrayTHW],
    *,
    cmap: str | Sequence[str] = "viridis",
    dpi: int = 150,
    figure_title: str | None = None,
    footer_text: str | None = None,
    fps: int = 2,
    group_axes: tuple[int, int] | None = None,
    panel_titles: Sequence[str] | None = None,
    vmax: float | Sequence[float | None] | None = None,
    vmin: float | Sequence[float | None] | None = None,
    video_format: Literal["mp4", "gif"] = "mp4",
) -> BytesIO:
    """Render multiple panels side-by-side, animated over time.

    Builds the first frame with `render_panels`, then updates each panel's
    image data per frame.

    Args:
        arrays: 1 to 3 3D `[T, H, W]` arrays, one per panel, animated in lockstep.
        cmap: Optional colourmap(s), either shared by all panels or one per panel.
        dpi: Dots per inch for the rendered video frames.
        figure_title: Optional figure-level title drawn above all panels.
        footer_text: Optional footer text drawn below the colourbars.
        fps: Frames per second for the rendered video.
        group_axes: Optional inclusive `(start, end)` panel index range to share a colourbar.
        panel_titles: Optional per-panel titles, one per panel.
        vmax: Optional upper colour-scale bound(s), either shared or one per panel.
        vmin: Optional lower colour-scale bound(s), either shared or one per panel.
        video_format: Encode as "mp4" or "gif".

    Returns:
        A BytesIO buffer containing the encoded video.

    """
    figure, axes = render_panels(
        [array[0] for array in arrays],
        cmap=cmap,
        figure_title=figure_title,
        footer_text=footer_text,
        group_axes=group_axes,
        panel_titles=panel_titles,
        vmax=vmax,
        vmin=vmin,
    )
    images = [ax.images[0] for ax in axes]

    def animate(tt: int) -> tuple[()]:
        for image, array in zip(images, arrays, strict=True):
            image.set_data(array[tt])
        return ()

    try:
        anim = animation.FuncAnimation(
            figure, animate, frames=arrays[0].shape[0], interval=1000 // fps
        )
        return video_from_animation(anim, dpi=dpi, fps=fps, video_format=video_format)
    finally:
        plt.close(figure)
