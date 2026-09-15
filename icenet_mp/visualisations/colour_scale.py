import logging

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm

from icenet_mp.types import DiffColourmap, DiffMode
from icenet_mp.utils import safe_nanmax, safe_nanmin

logger = logging.getLogger(__name__)


class ColourScale:
    """Builds matplotlib colour-mapping (norm, cmap, vmin/vmax) for a panel's array.

    Covers every case a rendered panel needs a colour scale for: a raw
    variable field (optionally centred, e.g. for a z-score panel), a
    ground-truth/prediction difference field, and NaN-safe colourmap
    lookup.
    """

    def __init__(self, diff_mode: DiffMode) -> None:
        """Initialise a ColourScale with a difference mode."""
        self._diff_mode = diff_mode

    def colourmap(
        self, cmap_name: str = "viridis", *, bad_color: str = "#dcdcdc"
    ) -> Colormap:
        """Create a colourmap copy with a specified color for bad (NaN) values.

        This function copies the specified colourmap and sets the 'bad' color to handle
        NaN values consistently, preventing white artifacts in visualisations.

        Args:
            cmap_name: Name of the matplotlib colourmap (e.g., "viridis", "RdBu_r").
                       Defaults to "viridis".
            bad_color: Color to use for NaN/bad values. Default is light grey (#dcdcdc).

        Returns:
            A copy of the colourmap with set_bad() configured.

        """
        cmap = mpl.colormaps.get_cmap(cmap_name)
        try:
            cmap = cmap.copy()
        except (AttributeError, TypeError):
            # Some matplotlib versions return non-copyable colourmap; create new
            cmap = mpl.colormaps.get_cmap(cmap.name)

        cmap.set_bad(bad_color)
        return cmap

    def diff_colourmap(
        self,
        sample: np.ndarray | float,
    ) -> DiffColourmap:
        """Construct colour mapping settings for a difference panel.

        Behaviour depends on the difference mode:

        - "signed": symmetric diverging scale centred on 0,
          useful for showing positive vs negative bias.
        - "absolute" / "smape": sequential scale from 0 to max,
          useful for showing error magnitude.

        Args:
            sample: Either a full array of differences (for precompute mode)
                    or a scalar maximum difference (for two-pass mode).

        Returns:
            DiffRenderParams: Normalisation, colour limits, and colourmap.

        """
        if self._diff_mode == "signed":
            # Force symmetric limits around zero so 0 is the literal midpoint
            if isinstance(sample, (float, int)):
                max_abs = max(1.0, float(abs(sample)))
                vmin, vmax = -max_abs, max_abs
            else:
                # Find the min and max values of the sample array using safe helpers
                vmin_data = safe_nanmin(sample, default=-1.0)
                vmax_data = safe_nanmax(sample, default=1.0)
                # Find the maximum absolute value of the sample array
                max_abs = max(1.0, abs(vmin_data), abs(vmax_data))
                vmin, vmax = -max_abs, max_abs

            return DiffColourmap(
                norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
                vmin=None,
                vmax=None,
                cmap="RdBu_r",
            )

        if self._diff_mode in ("absolute", "smape"):
            # Positive-only scale
            if isinstance(sample, (float, int)):
                vmax = max(1e-6, float(sample))
            else:
                vmax = max(1e-6, safe_nanmax(sample, default=0.0))

            return DiffColourmap(
                norm=None,
                vmin=0.0,
                vmax=vmax,
                cmap="magma",
            )

        msg = f"Unknown difference mode: {self._diff_mode}"
        raise ValueError(msg)

    def bounds(
        self, diff_colour_scale: DiffColourmap
    ) -> tuple[float | None, float | None]:
        """Resolve the effective (vmin, vmax) from a `DiffColourmap`.

        A diverging scale (mode "signed") carries its bounds on `norm`;
        a sequential scale (mode "absolute"/"smape") carries them directly
        as `vmin`/`vmax`. Callers that only need plain bounds (e.g. to hand
        to `Renderer`) shouldn't need to know which encoding `diff_colourmap()`
        chose.
        """
        if diff_colour_scale.norm is not None:
            return diff_colour_scale.norm.vmin, diff_colour_scale.norm.vmax
        return diff_colour_scale.vmin, diff_colour_scale.vmax

    def normalisation(
        self,
        data: np.ndarray,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        centre: float | None = None,
    ) -> Normalize:
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
            A Normalize (linear) or TwoSlopeNorm (centred) instance, with its
            vmin/vmax (and vcenter, if centred) already resolved.

        """
        # Compute data range with robust handling of NaN/inf
        data_min = safe_nanmin(data)
        data_max = safe_nanmax(data)

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
            return TwoSlopeNorm(vmin=final_vmin, vcenter=float(centre), vmax=final_vmax)

        # Linear colourmap
        final_vmin = float(vmin if vmin is not None else data_min)
        final_vmax = float(vmax if vmax is not None else data_max)
        return Normalize(vmin=final_vmin, vmax=final_vmax)
