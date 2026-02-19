from logging import getLogger

from icenet_mp.data_processors.filters import register_filters
from icenet_mp.data_processors.sources import register_sources
from icenet_mp.visualisations import register_animation_backends
from icenet_mp.xpu import register_accelerators

logger = getLogger(__name__)


def register_plugins() -> None:
    """Register all plugins."""
    logger.debug("Registering plugins with external libraries...")
    register_accelerators()
    register_animation_backends()
    register_filters()
    register_sources()
