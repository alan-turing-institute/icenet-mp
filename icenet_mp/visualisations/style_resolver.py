import logging
from collections.abc import Mapping
from typing import Any

from icenet_mp.types import ColourStyle

logger = logging.getLogger(__name__)


class StyleResolver:
    """Resolve variable style from a config."""

    def __init__(
        self, styles: Mapping[str, Mapping[str, Any]] | None, default_cmap: str
    ) -> None:
        """Construct a StyleResolver from a dict with a default colourmap."""
        self._styles = styles
        self._default_cmap = default_cmap

    @staticmethod
    def _normalise(name: str) -> str:
        """Normalise separators in a variable name so config keys match consistently.

        'era5__2t' and 'era5-2t' both normalise to 'era5:2t'.
        """
        # Convert double-underscore to colon (this maps 'era5__2t' -> 'era5:2t')
        name = name.replace("__", ":")
        # Treat hyphens as separators too: 'era5-2t' -> 'era5:2t'
        name = name.replace("-", ":")
        # Collapse accidental repeated '::' to single ':'
        while "::" in name:
            name = name.replace("::", ":")
        # Keep single underscores (they are meaningful in some variable names)
        return name

    def _match(self, var_name: str) -> Mapping[str, Any] | None:
        """Return the best matched style config for a variable from the bound dict.

        Matching priority:
          1) exact key (raw, then with separators normalised)
          2) wildcard prefix key ending with '*'
          3) _default
          4) no match (None)
        """
        if not self._styles or not isinstance(self._styles, Mapping):
            logger.debug(
                "StyleResolver: styles is a %s not a Mapping",
                type(self._styles),
            )
            return None

        norm_var = self._normalise(var_name)
        spec: Mapping[str, Any] | None
        for key in dict.fromkeys((var_name, norm_var)):
            spec = self._styles.get(key)
            if isinstance(spec, Mapping):
                return spec

        if (spec := self._wildcard_match(var_name, norm_var)) is not None:
            return spec

        spec = self._styles.get("_default", None)
        return spec if isinstance(spec, Mapping) else None

    def _wildcard_match(self, var_name: str, norm_var: str) -> Mapping[str, Any] | None:
        """Scan wildcard-suffixed keys (e.g. 'era5:*') for a prefix match."""
        if not self._styles:
            return None

        for key in self._styles:
            if not (isinstance(key, str) and key.endswith("*")):
                continue
            prefix = key[:-1]
            prefix_norm = self._normalise(prefix)
            # If prefix_norm is empty (user wrote '*' only) skip it
            if not prefix_norm:
                continue
            # Compare against both raw and normalised var names
            if var_name.startswith(prefix) or norm_var.startswith(prefix_norm):
                spec = self._styles.get(key, None)
                if isinstance(spec, Mapping):
                    return spec
        return None

    def style_for_variable(self, var_name: str) -> ColourStyle:
        """Resolve variable style, falling back to the default colourmap if needed."""
        spec = self._match(var_name) or {}
        return ColourStyle(
            cmap=spec.get("cmap") or self._default_cmap,
            vmin=spec.get("vmin"),
            vmax=spec.get("vmax"),
            units=spec.get("units"),
        )
