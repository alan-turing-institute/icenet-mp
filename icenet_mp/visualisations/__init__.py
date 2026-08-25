from imageio_ffmpeg import get_ffmpeg_exe
from matplotlib import rcParams

from .dataset_plotting import plot_variables_static, plot_variables_video
from .helpers import DEFAULT_SIC_SPEC
from .plotter import Plotter
from .plotting_core import compute_standardised_difference
from .plotting_static import plot_static_uncertainty


def register_animation_backends() -> None:
    """Register the ImageIO FFMPEG animation backend."""
    rcParams["animation.ffmpeg_path"] = get_ffmpeg_exe()


__all__ = [
    "DEFAULT_SIC_SPEC",
    "Plotter",
    "compute_standardised_difference",
    "plot_static_uncertainty",
    "plot_variables_static",
    "plot_variables_video",
]
