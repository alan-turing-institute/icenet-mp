from imageio_ffmpeg import get_ffmpeg_exe
from matplotlib import rcParams

from .dataset_plotting import plot_variables_static, plot_variables_video
from .helpers import DEFAULT_SIC_SPEC
from .plotter import Plotter


def register_animation_backends() -> None:
    """Register the ImageIO FFMPEG animation backend."""
    rcParams["animation.ffmpeg_path"] = get_ffmpeg_exe()


__all__ = [
    "DEFAULT_SIC_SPEC",
    "Plotter",
    "plot_variables_static",
    "plot_variables_video",
]
