import warnings
from logging import getLogger

from icenet_mp.compatibility.lightning import register_accelerators
from icenet_mp.compatibility.torch import patch_parameter_deepcopy
from icenet_mp.data_processors.filters import register_filters
from icenet_mp.data_processors.sources import register_sources
from icenet_mp.visualisations import register_animation_backends

log = getLogger(__name__)


def configure_external_libraries() -> None:
    """Configure any external libraries used by the application."""
    log.debug("Configuring external libraries...")
    patch_parameter_deepcopy()
    register_accelerators()
    register_animation_backends()
    register_filters()
    register_sources()
    # Ignore warnings about known PyTorch issues
    warnings.filterwarnings(
        "ignore",
        message=".*Using padding='same' with even kernel lengths and odd dilation.*",
    )


__all__ = ["configure_external_libraries"]
