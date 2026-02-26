import logging

from anemoi.transform.filters import filter_registry

from .doubling_filter import DoublingFilter
from .nan_to_num import NanToNum

logger = logging.getLogger(__name__)


def register_filters() -> None:
    """Register all filters with anemoi-transform."""
    filters = {
        "doubling_filter": DoublingFilter,
        "nan_to_num": NanToNum,
    }
    for filter_name, filter_class in filters.items():
        if filter_name not in filter_registry.registered:
            filter_registry.register(filter_name, filter_class)
            logger.debug("Registered %s with anemoi-transform.", filter_class.__name__)


__all__ = [
    "DoublingFilter",
    "NanToNum",
    "register_filters",
]
