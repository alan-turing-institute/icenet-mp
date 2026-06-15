import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm

from icenet_mp.types import DiffColourmapSpec, DiffMode, DiffStrategy, PlotSpec

logger = logging.getLogger(__name__)


@dataclass
class VariableStyle:
    """Styling configuration for individual variables.

    Attributes:
        cmap: Matplotlib colourmap name (e.g., "viridis", "RdBu_r").
        colourbar_strategy: "shared" or "separate" (kept for compatibility).
        vmin: Minimum value for colour scale.
        vmax: Maximum value for colour scale.
        two_slope_centre: Centre value for diverging colourmap (TwoSlopeNorm).
        units: Display units for the variable (e.g., "K", "m/s").
        origin: Imshow origin override ("upper" keeps north-up, "lower" keeps south-up).
        decimals: Number of decimal places for colourbar tick labels (default: 2).
        use_scientific_notation: Whether to format colourbar tick labels in scientific notation (default: False).

    """

    cmap: str | None = None
    colourbar_strategy: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    two_slope_centre: float | None = None
    units: str | None = None
    origin: Literal["upper", "lower"] | None = None
    decimals: int | None = None
    use_scientific_notation: bool | None = None


def colourmap_with_bad(cmap_name: str | None, bad_color: str = "#dcdcdc") -> Colormap:
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


def safe_nanmin(arr: np.ndarray, default: float = 0.0) -> float:
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


def safe_nanmax(arr: np.ndarray, default: float = 1.0) -> float:
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
    var_name: str, styles: dict[str, dict[str, Any]] | None
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
        return VariableStyle(**{k: spec.get(k) for k in VariableStyle.__annotations__})

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
        return VariableStyle(**{k: spec.get(k) for k in VariableStyle.__annotations__})

    return VariableStyle()


def levels_from_spec(spec: PlotSpec) -> np.ndarray:
    """Generate contour levels from a plotting specification.

    Creates evenly-spaced contour levels based on the minimum and maximum values
    specified in the PlotSpec. If vmin or vmax are not specified, defaults to
    0.0 and 1.0.

    Args:
        spec: PlotSpec object containing vmin and vmax parameters.

    Returns:
        NumPy array of contour levels spanning from vmin to vmax with
        n_contour_levels steps.

    """
    vmin = 0.0 if spec.vmin is None else spec.vmin
    vmax = 1.0 if spec.vmax is None else spec.vmax
    return np.linspace(vmin, vmax, spec.n_contour_levels)


def create_normalisation(
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

    norm_linear: Normalize | TwoSlopeNorm = Normalize(vmin=final_vmin, vmax=final_vmax)
    return norm_linear, final_vmin, final_vmax


def compute_difference(
    ground_truth: np.ndarray, prediction: np.ndarray, diff_mode: DiffMode
) -> np.ndarray:
    """Compute the difference between the ground truth and prediction.

    Args:
        ground_truth: The ground truth array. [T,H,W]
        prediction: The prediction array. [T,H,W]
        diff_mode: Method to compute the difference.

    Returns:
        Difference array. [T,H,W]

    Raises:
        ValueError: If the difference mode is invalid.

    """
    if diff_mode == "signed":
        return ground_truth - prediction
    if diff_mode == "absolute":
        return np.abs(ground_truth - prediction)
    if diff_mode == "smape":
        denom = np.clip((np.abs(ground_truth) + np.abs(prediction)) / 2.0, 1e-6, None)
        return np.abs(prediction - ground_truth) / denom
    msg = f"Invalid difference mode: {diff_mode}"
    raise ValueError(msg)


def make_diff_colourmap(
    sample: np.ndarray | float,
    *,
    mode: DiffMode,
) -> DiffColourmapSpec:
    """Construct colour mapping settings for a difference panel.

    Behaviour depends on the difference mode:

    - "signed": symmetric diverging scale centred on 0,
      useful for showing positive vs negative bias.
    - "absolute" / "smape": sequential scale from 0 to max,
      useful for showing error magnitude.

    Args:
        sample: Either a full array of differences (for precompute mode)
                or a scalar maximum difference (for two-pass mode).
        mode: Difference mode ("signed", "absolute", or "smape").

    Returns:
        DiffRenderParams: Normalisation, colour limits, and colourmap.

    """
    if mode == "signed":
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

        return DiffColourmapSpec(
            norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
            vmin=None,
            vmax=None,
            cmap="RdBu_r",
        )

    if mode in ("absolute", "smape"):
        # Positive-only scale
        if isinstance(sample, (float, int)):
            vmax = max(1e-6, float(sample))
        else:
            vmax = max(1e-6, safe_nanmax(sample, default=0.0))

        return DiffColourmapSpec(
            norm=None,
            vmin=0.0,
            vmax=vmax,
            cmap="magma",
        )

    msg = f"Unknown difference mode: {mode}"
    raise ValueError(msg)


def prepare_difference_stream(
    *,
    include_difference: bool,
    diff_mode: DiffMode,
    strategy: DiffStrategy,
    ground_truth_stream: np.ndarray,
    prediction_stream: np.ndarray,
) -> tuple[np.ndarray | None, DiffColourmapSpec | None]:
    """General, reusable planner for animations (maps or time-series).

    This function implements three different strategies for handling
    difference between ground truth and prediction in animations, each with
    different memory and computational trade-offs. The choice of strategy affects
    both memory usage and animation performance.

    Args:
        include_difference: Whether difference visualisation is requested.
        diff_mode: Type of difference computation (signed/absolute/smape).
        strategy: Strategy for difference computation:
            - "precompute": Calculate all differences upfront
            - "two-pass": Scan data to determine colour scale, then compute per-frame
            - "per-frame": Compute differences on-demand
        ground_truth_stream: 3D array of ground truth data over time.
        prediction_stream: 3D array of prediction data over time.

    Returns:
        Tuple of (difference_stream, colour_scale):
        - difference_stream: None unless strategy == 'precompute'
        - colour_scale: DiffColourmapSpec describing colourmap/norm/range

    """
    if not include_difference:
        return None, None

    n_timesteps = ground_truth_stream.shape[0]
    if strategy == "precompute":
        difference_stream = compute_difference(
            ground_truth_stream, prediction_stream, diff_mode
        )
        colour_scale = make_diff_colourmap(difference_stream, mode=diff_mode)
        return difference_stream, colour_scale

    if strategy == "two-pass":
        differences = [
            compute_difference(
                ground_truth_stream[tt], prediction_stream[tt], diff_mode
            )
            for tt in range(n_timesteps)
        ]
        max_ = max(
            float(
                np.nanmax(np.abs(difference) if diff_mode == "signed" else difference)
                or 0.0
            )
            for difference in differences
        )
        colour_scale = make_diff_colourmap(max_, mode=diff_mode)
        return None, colour_scale

    if strategy == "per-frame":
        # no precomputation; each frame will call compute_difference and
        # may choose to infer its own params if desired (less consistent look)
        return None, None

    msg = f"Unknown DiffStrategy: {strategy}"
    raise ValueError(msg)


def compute_display_ranges(
    ground_truth: np.ndarray, prediction: np.ndarray, plot_spec: PlotSpec
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute vmin/vmax for ground truth and prediction based on strategy.

    Args:
        ground_truth: The ground truth array. [H,W]
        prediction: The prediction array. [H,W]
        plot_spec: The plotting specification.

    Returns:
        The display ranges. (vmin, vmax)

    Raises:
        InvalidArrayError: If the arrays are not 2D or have different shapes.

    """
    # Get data ranges for ground_truth and prediction
    groundtruth_min = float(np.nanmin(ground_truth))
    groundtruth_max = float(np.nanmax(ground_truth))
    prediction_min = float(np.nanmin(prediction))
    prediction_max = float(np.nanmax(prediction))

    # Both panels use the same vmin/vmax (ground truth used as reference)
    if plot_spec.colourbar_strategy == "shared":
        shared_range = (groundtruth_min, groundtruth_max)
        return shared_range, shared_range

    # Each panel uses its own data range
    if plot_spec.colourbar_strategy == "separate":
        return (groundtruth_min, groundtruth_max), (prediction_min, prediction_max)

    # Fallback - should not reach here but required for type checking
    default_range = (
        plot_spec.vmin if plot_spec.vmin is not None else 0.0,
        plot_spec.vmax if plot_spec.vmax is not None else 1.0,
    )
    return default_range, default_range
