import logging
from collections.abc import Mapping
from typing import Any

from icenet_mp.types import ColourScale

log = logging.getLogger(__name__)


class StyleResolver:
    """Resolve variable style from a config."""

    def __init__(
        self, styles: Mapping[str, Mapping[str, Any]] | None, default_cmap: str
    ) -> None:
        """Construct a StyleResolver from a dict with a default colourmap."""
        if not isinstance(styles, Mapping):
            log.warning("styles is a %s not a Mapping", type(styles))
            styles = {}
        self._styles = styles
        self._default_cmap = default_cmap

    def _match(self, var_name: str) -> Mapping[str, Any] | None:
        """Return the best matched style config for a variable from the bound dict.

        Matching priority:
          1) exact key
          2) wildcard prefix key ending with '*'
          3) _default
          4) no match (None)
        """
        candidates = (
            self._styles.get(var_name),
            self._wildcard_match(var_name),
            self._styles.get("_default"),
        )
        return next((spec for spec in candidates if isinstance(spec, Mapping)), None)

    def _wildcard_match(self, var_name: str) -> Mapping[str, Any] | None:
        """Scan wildcard-suffixed keys (e.g. 'era5:*') for a prefix match."""
        for key, spec in self._styles.items():
            if not (isinstance(key, str) and key.endswith("*")):
                continue
            prefix = key[:-1]
            # If prefix is empty (user wrote '*' only) skip it
            if prefix and var_name.startswith(prefix) and isinstance(spec, Mapping):
                return spec
        return None

    def colour_scale(self, var_name: str) -> ColourScale:
        """Construct a ColourScale for this variable, with a default cmap fallback."""
        spec = self._match(var_name) or {}
        return ColourScale(
            cmap=spec.get("cmap") or self._default_cmap,
            vmin=spec.get("vmin"),
            vmax=spec.get("vmax"),
            units=spec.get("units"),
        )
