import logging
from collections.abc import Mapping
from typing import Any

from icenet_mp.types import VariableStyle

logger = logging.getLogger(__name__)


class VariableStyleResolver:
    """Resolve variable style from a config."""

    def __init__(
        self, styles: dict[str, dict[str, Any]] | None, default_cmap: str
    ) -> None:
        """Bind the styles dict and the colourmap to fall back to when unstyled."""
        self._styles = styles
        self._default_cmap = default_cmap

    def _match(  # noqa: C901, PLR0911
        self, var_name: str
    ) -> Mapping[str, Any] | None:
        """Return the best matching style config for a variable from the bound styles dict.

        Matching priority:
          1) exact key
          2) wildcard prefix key ending with '*'
          3) _default
          4) no match (None)
        Accepts any Mapping (so OmegaConf DictConfig works).
        """

        def _normalise_name(name: str) -> str:
            # Convert double-underscore to colon (this maps 'era5__2t' -> 'era5:2t')
            name = name.replace("__", ":")
            # Treat hyphens as separators too: 'era5-2t' -> 'era5:2t'
            name = name.replace("-", ":")
            # Collapse accidental repeated '::' to single ':'
            while "::" in name:
                name = name.replace("::", ":")
            # Keep single underscores (they are meaningful in some variable names)
            return name

        if not self._styles:
            return None

        # Accept Mapping-like configs (Dict, DictConfig, etc.)
        if not isinstance(self._styles, Mapping):
            logger.info("style_for_variable: styles is not a Mapping; ignoring styles")
            return None

        # Quick exact match first (try raw var_name)
        spec = self._styles.get(var_name)
        if isinstance(spec, Mapping):
            return spec

        # Try normalised exact match
        norm_var = _normalise_name(var_name)
        if norm_var != var_name:
            spec = self._styles.get(norm_var)
            if isinstance(spec, Mapping):
                return spec

        # Wildcard prefix match: scan keys ending with '*' (normalise the key before comparing)
        # We iterate keys so keep original order (OmegaConf preserves insertion order).
        for key in self._styles:
            if isinstance(key, str) and key.endswith("*"):
                prefix = key[:-1]
                prefix_norm = _normalise_name(prefix)
                # If prefix_norm is empty (user wrote '*' only) skip it
                if not prefix_norm:
                    continue
                # Compare against both raw and normalised var names
                if var_name.startswith(prefix) or norm_var.startswith(prefix_norm):
                    spec = self._styles.get(key)
                    if isinstance(spec, Mapping):
                        return spec
                    logger.info(
                        "style_for_variable: wildcard candidate %r not a dict (type=%s)",
                        key,
                        type(spec),
                    )

        # Fallback to _default
        spec = self._styles.get("_default")
        if isinstance(spec, Mapping):
            return spec

        return None

    def style_for_variable(self, var_name: str) -> VariableStyle:
        """Return the resolved style for a variable, with cmap always set.

        Delegates matching to `_match`, which returns the raw matched config
        (or None); this is the single place that turns that into a
        `VariableStyle`, so callers never need their own `style.cmap or
        default` fallback -- an unmatched or unset cmap falls back to the
        colourmap bound at construction.
        """
        spec = self._match(var_name) or {}
        return VariableStyle(
            cmap=spec.get("cmap") or self._default_cmap,
            vmin=spec.get("vmin"),
            vmax=spec.get("vmax"),
            units=spec.get("units"),
        )
