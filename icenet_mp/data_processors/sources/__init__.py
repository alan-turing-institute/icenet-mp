import logging
from typing import Annotated, Union

import anemoi.datasets.create.recipe.action as _action
from anemoi.datasets.create.recipe import Recipe
from anemoi.datasets.create.recipe.action import (
    _action_discriminator as _discriminator,
)
from anemoi.datasets.create.recipe.action import (
    _factories,
    _schemas,
)
from anemoi.datasets.create.sources import source_registry
from pydantic import Discriminator

from .argo import ArgoSource
from .ftp import FTPSource

logger = logging.getLogger(__name__)


def register_sources() -> None:
    """Rebuild anemoi's Recipe validation to include icenet-mp sources.

    Although our custom sources are registered with the source registry at import time,
    the Recipe Action union is only built once at module load time. To force a rebuild
    we clear cached factories and schemas, reset the Action union then rebuild Recipe.
    """
    # Perform an idempotent source registration
    sources = {
        "ftp": FTPSource,
        "argo": ArgoSource,
    }
    for name, source in sources.items():
        if name not in source_registry.registered:
            source_registry.register(name, source)
            logger.debug("Registered %s with anemoi-datasets.", source.__name__)

    # Also reset the Pydantic model for Recipe which has a list of allowed sources baked
    # into it
    _factories.cache_clear()
    _schemas.cache_clear()
    _action.Action = Annotated[
        Union[*_schemas()],
        Discriminator(_discriminator),
    ]
    Recipe.model_rebuild(force=True)


__all__ = [
    "ArgoSource",
    "FTPSource",
    "register_sources",
]
