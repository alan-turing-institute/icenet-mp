import contextlib
import gc
import io
import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from matplotlib import animation
from matplotlib.figure import Figure
from PIL import Image
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import VideoRenderError


@contextlib.contextmanager
def _suppress_mpl_animation_logs() -> Iterator[None]:
    """Temporarily suppress matplotlib animation INFO log messages."""
    mpl_logger = logging.getLogger("matplotlib.animation")
    original_level = mpl_logger.level
    try:
        mpl_logger.setLevel(logging.WARNING)
        yield
    finally:
        mpl_logger.setLevel(original_level)


def image_from_figure(fig: Figure, *, dpi: int) -> ImageFile:
    """Convert a matplotlib figure to a PIL image file."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


def video_from_animation(
    anim: animation.FuncAnimation,
    *,
    dpi: int = 200,
    fps: int = 2,
    video_format: Literal["mp4", "gif"] = "gif",
) -> io.BytesIO:
    """Save an animation to a temporary file and return BytesIO (with cleanup)."""
    suffix = ".gif" if video_format.lower() == "gif" else ".mp4"

    writer: animation.AbstractMovieWriter | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            try:
                # Save video to tempfile
                writer = (
                    animation.PillowWriter(fps=fps)
                    if suffix == ".gif"
                    else animation.FFMpegWriter(
                        fps=fps,
                        codec="libx264",
                        bitrate=1800,
                        # Ensure dimensions are compatible with yuv420p (even width/height)
                        # by applying a scale filter that truncates to the nearest even integers.
                        extra_args=[
                            "-pix_fmt",
                            "yuv420p",
                            "-vf",
                            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        ],
                    )
                )
                # Suppress matplotlib's INFO log message about writer selection
                with _suppress_mpl_animation_logs():
                    anim.save(tmp.name, writer=writer, dpi=dpi)
                # Load tempfile into a BytesIO buffer
                with Path(tmp.name).open("rb") as fh:
                    buffer = io.BytesIO(fh.read())
            except (OSError, MemoryError) as err:
                msg = f"Video encoding failed: {err!s}"
                raise VideoRenderError(msg) from err
    finally:
        # Explicitly cleanup writer to prevent semaphore leaks
        if writer is not None and hasattr(writer, "cleanup"):
            with contextlib.suppress(OSError, RuntimeError, AttributeError):
                writer.cleanup()
        # Force garbage collection to clean up any remaining resources
        gc.collect()

    buffer.seek(0)
    return buffer
