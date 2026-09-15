import logging
from collections.abc import Mapping
from typing import Any

from icenet_mp.types import VariableStyle

logger = logging.getLogger(__name__)


class VariableStyleResolver:
    """Resolves a variable's declared display style from config."""

    def style_for_variable(  # noqa: C901, PLR0911
        self, var_name: str, styles: dict[str, dict[str, Any]] | None
    ) -> VariableStyle:
        """Return best matching style for a variable from config styles dict.

        Matching priority:
          1) exact key
          2) wildcard prefix key ending with '*'
          3) _default
          4) empty style
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

        if not styles:
            return VariableStyle()

        # Accept Mapping-like configs (Dict, DictConfig, etc.)
        if not isinstance(styles, Mapping):
            logger.info("style_for_variable: styles is not a Mapping; ignoring styles")
            return VariableStyle()

        # Quick exact match first (try raw var_name)
        spec = styles.get(var_name)
        if isinstance(spec, Mapping):
            return VariableStyle(
                **{k: spec.get(k) for k in VariableStyle.__annotations__}
            )

        # Try normalised exact match
        norm_var = _normalise_name(var_name)
        if norm_var != var_name:
            spec = styles.get(norm_var)
            if isinstance(spec, Mapping):
                return VariableStyle(
                    **{k: spec.get(k) for k in VariableStyle.__annotations__}
                )

        # Wildcard prefix match: scan keys ending with '*' (normalise the key before comparing)
        # We iterate keys so keep original order (OmegaConf preserves insertion order).
        for key in styles:
            if isinstance(key, str) and key.endswith("*"):
                prefix = key[:-1]
                prefix_norm = _normalise_name(prefix)
                # If prefix_norm is empty (user wrote '*' only) skip it
                if not prefix_norm:
                    continue
                # Compare against both raw and normalised var names
                if var_name.startswith(prefix) or norm_var.startswith(prefix_norm):
                    spec = styles.get(key)
                    if isinstance(spec, Mapping):
                        return VariableStyle(
                            **{k: spec.get(k) for k in VariableStyle.__annotations__}
                        )
                    logger.info(
                        "style_for_variable: wildcard candidate %r not a dict (type=%s)",
                        key,
                        type(spec),
                    )

        # Fallback to _default
        spec = styles.get("_default")
        if isinstance(spec, Mapping):
            return VariableStyle(
                **{k: spec.get(k) for k in VariableStyle.__annotations__}
            )

        return VariableStyle()
