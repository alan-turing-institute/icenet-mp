import logging
from collections.abc import Mapping
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm

from icenet_mp.types import VariableStyle

logger = logging.getLogger(__name__)


class VariableStyler:
    """Resolves per-variable styling and turns a single array into a coloured, normalised image."""

    def colourmap_with_bad(
        self, cmap_name: str | None, bad_color: str = "#dcdcdc"
    ) -> Colormap:
        """Create a colourmap copy with a specified color for bad (NaN) values.

        This function copies the specified colourmap and sets the 'bad' color to handle
        NaN values consistently, preventing white artifacts in visualisations.

        Args:
            cmap_name: Name of the matplotlib colourmap (e.g., "viridis", "RdBu_r").
                       If None, defaults to "viridis".
            bad_color: Color to use for NaN/bad values. Default is light grey (#dcdcdc).

        Returns:
            A copy of the colourmap with set_bad() configured.

        """
        if cmap_name is None:
            cmap = mpl.colormaps.get_cmap("viridis")
        else:
            cmap = mpl.colormaps.get_cmap(cmap_name)

        try:
            cmap = cmap.copy()
        except (AttributeError, TypeError):
            # Some matplotlib versions return non-copyable colourmap; create new
            cmap = mpl.colormaps.get_cmap(cmap.name)

        cmap.set_bad(bad_color)
        return cmap

    def safe_nanmin(self, arr: np.ndarray, default: float = 0.0) -> float:
        """Safely compute nanmin with fallback for empty or all-NaN arrays.

        Args:
            arr: Array to compute minimum from.
            default: Default value if array is empty or all NaN.

        Returns:
            Minimum value or default.

        """
        if np.isfinite(arr).any():
            result = np.nanmin(arr)
            return float(result) if np.isfinite(result) else default
        return default

    def safe_nanmax(self, arr: np.ndarray, default: float = 1.0) -> float:
        """Safely compute nanmax with fallback for empty or all-NaN arrays.

        Args:
            arr: Array to compute maximum from.
            default: Default value if array is empty or all NaN.

        Returns:
            Maximum value or default.

        """
        if np.isfinite(arr).any():
            result = np.nanmax(arr)
            return float(result) if np.isfinite(result) else default
        return default

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

    def create_normalisation(
        self,
        data: np.ndarray,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        centre: float | None = None,
    ) -> tuple[Normalize | TwoSlopeNorm, float, float]:
        """Create appropriate normalisation for data with optional centring.

        This function creates either a linear Normalize or a diverging TwoSlopeNorm
        based on whether a centre value is provided. When a centre is specified,
        the normalisation will be symmetric around that value.

        Args:
            data: 2D array of data to normalise.
            vmin: Minimum value for colour scale. If None, inferred from data.
            vmax: Maximum value for colour scale. If None, inferred from data.
            centre: Centre value for diverging colourmap. If provided, creates
                a symmetric TwoSlopeNorm around this value.

        Returns:
            Tuple of (normalisation, vmin, vmax) where:
            - normalisation: Normalize or TwoSlopeNorm object
            - vmin: Computed minimum value
            - vmax: Computed maximum value

        """
        # Compute data range with robust handling of NaN/inf
        data_min = float(np.nanmin(data)) if np.isfinite(data).any() else 0.0
        data_max = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0

        if centre is not None:
            # Diverging colourmap centred at specified value
            low = vmin if vmin is not None else data_min
            high = vmax if vmax is not None else data_max

            # Make symmetric around the centre where possible
            span_low = abs(centre - low)
            span_high = abs(high - centre)
            span = max(span_low, span_high, 1e-6)

            final_vmin = float(centre - span)
            final_vmax = float(centre + span)

            norm: Normalize | TwoSlopeNorm = TwoSlopeNorm(
                vmin=final_vmin, vcenter=float(centre), vmax=final_vmax
            )
            return norm, final_vmin, final_vmax

        # Linear colourmap
        final_vmin = float(vmin if vmin is not None else data_min)
        final_vmax = float(vmax if vmax is not None else data_max)

        norm_linear: Normalize | TwoSlopeNorm = Normalize(
            vmin=final_vmin, vmax=final_vmax
        )
        return norm_linear, final_vmin, final_vmax
