"""Minimal panel-plot and panel-video rendering core."""

from collections.abc import Sequence
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from PIL.ImageFile import ImageFile

from icenet_mp.types import ArrayHW, ArrayTHW

from .convert import image_from_figure, video_from_animation


def render_panels(
    arrays: Sequence[ArrayHW],
    *,
    titles: Sequence[str] | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 150,
) -> ImageFile:
    """Draw 1-3 side-by-side panels sharing one colourbar.

    Args:
        arrays: 1 to 3 2D `[H, W]` arrays, one per panel.
        titles: Optional per-panel title, same length as `arrays`.
        cmap: Colourmap shared by all panels.
        vmin: Lower colour-scale bound shared by all panels (None = infer).
        vmax: Upper colour-scale bound shared by all panels (None = infer).
        dpi: Dots per inch for the rendered image.

    Returns:
        A PIL image of the rendered panels.

    """
    fig, axes = plt.subplots(
        1, len(arrays), figsize=(5 * len(arrays), 5), layout="constrained"
    )
    axes = np.atleast_1d(axes)

    images = [
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        for ax, arr in zip(axes, arrays, strict=True)
    ]
    for ax, title in zip(axes, titles or [""] * len(arrays), strict=True):
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(images[0], ax=axes.tolist())

    try:
        return image_from_figure(fig, dpi=dpi)
    finally:
        plt.close(fig)


def render_panels_video(  # noqa: PLR0913
    arrays: Sequence[ArrayTHW],
    *,
    titles: Sequence[str] | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 150,
    fps: int = 2,
) -> BytesIO:
    """Draw 1-3 side-by-side panels sharing one colourbar, animated over time.

    Args:
        arrays: 1 to 3 3D `[T, H, W]` arrays, one per panel, animated in lockstep.
        titles: Optional per-panel title, same length as `arrays`.
        cmap: Colourmap shared by all panels.
        vmin: Lower colour-scale bound shared by all panels (None = infer).
        vmax: Upper colour-scale bound shared by all panels (None = infer).
        dpi: Dots per inch for the rendered video frames.
        fps: Frames per second for the rendered video.

    Returns:
        A BytesIO buffer containing the encoded video.

    """
    fig, axes = plt.subplots(
        1, len(arrays), figsize=(5 * len(arrays), 5), layout="constrained"
    )
    axes = np.atleast_1d(axes)

    images = [
        ax.imshow(arr[0], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        for ax, arr in zip(axes, arrays, strict=True)
    ]
    for ax, title in zip(axes, titles or [""] * len(arrays), strict=True):
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(images[0], ax=axes.tolist())

    n_frames = arrays[0].shape[0]

    def animate(tt: int) -> tuple[()]:
        for image, arr in zip(images, arrays, strict=True):
            image.set_data(arr[tt])
        return ()

    try:
        anim = animation.FuncAnimation(
            fig, animate, frames=n_frames, interval=1000 // fps
        )
        return video_from_animation(anim, dpi=dpi, fps=fps, video_format="mp4")
    finally:
        plt.close(fig)
