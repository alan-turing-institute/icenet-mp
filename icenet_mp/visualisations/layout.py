"""Plot Map Layout Module.

Handles the layout of the plot panels for maps and colourbars.
The module is primarily used through the plotting system to create comparison visualisations
with 2-3 panels (ground truth, prediction, optional difference).

Main Components:
    - Layout constants for consistent spacing and proportions
    - GridSpec-based layout system with configurable margins and gutters
    - Colourbar positioning (vertical/horizontal)
    - Figure sizing based on data dimensions and aspect ratio for the plot panels
    - Specialised tick formatting for colourbars

"""

import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from icenet_mp.types import PlotSpec

from .plotting_core import DiffColourmapSpec

logger = logging.getLogger(__name__)

# Panel index constants (used for layout logic)
PREDICTION_PANEL_INDEX = 1
DIFFERENCE_PANEL_INDEX = 2
MIN_PANEL_COUNT_FOR_PREDICTION = 2
MIN_PANEL_COUNT_FOR_DIFFERENCE = 3

# Horizontal colourbar inset constants
HCBAR_WIDTH_FRAC = 0.75  # centred 75% width
HCBAR_LEFT_FRAC = (1.0 - HCBAR_WIDTH_FRAC) / 2.0

# Small epsilon for numerical stability
EPSILON_SMALL = 1e-6

# --- Layout Configuration Dataclasses ---


@dataclass(frozen=True)
class ColourbarConfig:
    """Colourbar sizing and clamp behaviour (fractions of figure).

    Example:
        # Wider colourbar with higher max fraction
        cbar_config = ColourbarConfig(default_width_frac=0.08, max_fraction_of_fig=0.15)

    """

    default_width_frac: float = (
        0.06  # Width allocated for colourbar slots (fraction of panel width)
    )
    min_width_frac: float = 0.01  # Minimum colourbar width clamp
    max_width_frac: float = 0.5  # Maximum colourbar width clamp
    desired_physical_width_in: float = (
        0.45  # Desired physical colourbar width in inches
    )
    max_fraction_of_fig: float = 0.12  # Maximum colourbar width as fraction of figure
    default_height_frac: float = 0.07  # Thickness of horizontal colourbar row
    default_pad_frac: float = 0.03  # Vertical gap between plots and colourbar row

    def clamp_frac(self, frac: float, fig_width_in: float) -> float:
        """Return a sane clamped fraction for the cbar slot based on figure width.

        Computes a physical-limit fraction derived from desired physical width,
        then clamps the input fraction between min/max bounds.
        """
        # Compute a physical-limit fraction derived from desired physical width
        phys_frac = min(
            self.max_fraction_of_fig,
            self.desired_physical_width_in / max(fig_width_in, EPSILON_SMALL),
        )
        return float(
            max(self.min_width_frac, min(frac, phys_frac, self.max_width_frac))
        )


@dataclass(frozen=True)
class TitleFooterConfig:
    """Title and footer spacing, positioning, and styling."""

    title_space: float = 0.07  # Fraction of figure height reserved for title
    footer_space: float = 0.08  # Fraction of figure height reserved for footer
    title_fontsize: int = 12  # Font size for title
    footer_fontsize: int = 11  # Font size for footer and badge
    title_y: float = 0.98  # Y position for title (near top, in figure coordinates)
    footer_y: float = 0.03  # Y position for footer (near bottom, in figure coordinates)
    bbox_pad_title: float = 2.0  # Padding for title bbox
    bbox_pad_badge: float = 1.5  # Padding for badge bbox
    zorder_high: int = 1000  # High z-order for text overlays


@dataclass(frozen=True)
class GapConfig:
    """Aspect-aware physical gap settings for single-panel layouts."""

    base: float = 0.15  # Preferred gap for aspect≈1 panels
    min_val: float = 0.10  # Hard minimum
    max_val: float = 0.22  # Hard maximum
    wide_limit: float = 4.0  # Aspect ratio at which we reach the "wide" gap
    tall_limit: float = 0.5  # Aspect ratio at which we reach the "tall" gap
    wide_gap: float = 0.11  # Gap to use for very wide panels
    tall_gap: float = 0.19  # Gap to use for very tall panels


@dataclass(frozen=True)
class SinglePanelSpacing:
    """Encapsulate spacing controls for standalone panels."""

    gap: GapConfig = field(default_factory=GapConfig)
    outer_buffer_in: float = 0.16  # Padding outside map/cbar (both sides)
    edge_guard_in: float = 0.25  # Minimum blank space beyond colourbar for ticks
    right_margin_scale: float = 0.5  # Scale factor for right margin adjustment
    right_margin_offset: float = 0.02  # Offset for right margin adjustment


@dataclass(frozen=True)
class FormattingConfig:
    """Tick formatting and fallback value constants."""

    num_ticks_linear: int = 5  # Number of ticks for linear colourbars
    midpoint_factor: float = (
        0.5  # Factor for calculating midpoint values in symmetric ticks
    )
    default_vmin_fallback: float = 0.0  # Fallback minimum value
    default_vmax_fallback: float = 1.0  # Fallback maximum value
    default_vmin_diff_fallback: float = -1.0  # Fallback minimum for difference plots
    default_vmax_diff_fallback: float = 1.0  # Fallback maximum for difference plots


@dataclass(frozen=True)
class LayoutConfig:
    """Top-level layout tuning surface - passed into build functions.

    Groups all layout-related configuration into a single, discoverable object.
    This makes it easy to override defaults for testing or custom layouts.

    Example:
        # Wider colourbar and less title space
        my_layout = LayoutConfig(
            colourbar=ColourbarConfig(default_width_frac=0.08),
            title_footer=TitleFooterConfig(title_space=0.04)
        )
        fig, axs, cax = build_single_panel_figure(
            height=200, width=300, layout_config=my_layout, colourbar_location="vertical"
        )

    """

    base_height_in: float = 6.0  # Standard figure height in inches
    outer_margin: float = 0.05  # Outer margin around entire figure (prevents clipping)
    gutter_vertical: float = 0.03  # Default for side-by-side with vertical bars
    gutter_horizontal: float = 0.03  # Smaller gaps when colourbar is below
    colourbar: ColourbarConfig = field(default_factory=ColourbarConfig)
    title_footer: TitleFooterConfig = field(default_factory=TitleFooterConfig)
    single_panel_spacing: SinglePanelSpacing = field(default_factory=SinglePanelSpacing)
    formatting: FormattingConfig = field(default_factory=FormattingConfig)
    min_plot_fraction: float = (
        0.2  # Minimum plot width/height as fraction of available space
    )
    min_usable_height_fraction: float = (
        0.6  # Minimum fraction of figure height for plotting area
    )
    default_figsizes: dict[int, tuple[float, float]] = field(
        default_factory=lambda: {
            1: (8, 6),  # Single panel
            2: (12, 6),  # Ground truth + Prediction
            3: (18, 6),  # Ground truth + Prediction + Difference
        }
    )


# Default layout configuration instance (used when layout_config is None)
_DEFAULT_LAYOUT_CONFIG = LayoutConfig()

# --- Main Layout Functions ---


def build_single_panel_figure(  # noqa: PLR0913, PLR0915
    *,
    height: int,
    width: int,
    layout_config: LayoutConfig | None = None,
    colourbar_location: Literal["vertical", "horizontal"] = "vertical",
    cbar_width: float | None = None,
    cbar_height: float | None = None,
    cbar_pad: float | None = None,
) -> tuple[Figure, Axes, Axes]:
    """Create a single-panel figure with layout consistent with multi-panel maps.

    Args:
        height: Data height in pixels.
        width: Data width in pixels.
        layout_config: Optional LayoutConfig instance to override defaults.
                      If None, uses default configuration.
        colourbar_location: Orientation of colourbar ("vertical" or "horizontal").
        cbar_width: Optional override for colourbar width fraction.
        cbar_height: Optional override for colourbar height fraction (horizontal only).
        cbar_pad: Optional override for colourbar padding.

    """
    if height <= 0 or width <= 0:
        msg = "height and width must be positive for single panel layout."
        raise ValueError(msg)

    layout = layout_config or _DEFAULT_LAYOUT_CONFIG
    outer_margin = layout.outer_margin
    title_space = layout.title_footer.title_space
    footer_space = layout.title_footer.footer_space

    base_h = layout.base_height_in
    aspect = width / max(1, height)
    fig_w = base_h * aspect
    fig = plt.figure(
        figsize=(fig_w, base_h), constrained_layout=False, facecolor="none"
    )

    top_val = max(layout.min_usable_height_fraction, 1.0 - (outer_margin + title_space))
    bottom_val = outer_margin + footer_space
    usable_height = top_val - bottom_val

    if colourbar_location == "vertical":
        # --- Interpret cbar_width as "fraction of panel width" ---
        cw_panel = float(
            layout.colourbar.default_width_frac if cbar_width is None else cbar_width
        )
        fig_w_in = float(fig.get_size_inches()[0])
        cw_panel = layout.colourbar.clamp_frac(cw_panel, fig_w_in)

        spacing = layout.single_panel_spacing

        extra_left_frac = spacing.outer_buffer_in / max(fig_w_in, EPSILON_SMALL)
        extra_right_frac = spacing.outer_buffer_in / max(fig_w_in, EPSILON_SMALL)

        if cbar_pad is None:
            cbar_pad_inches = _default_vertical_gap_inches(aspect, spacing.gap)
            logger.debug(
                "single-panel vertical gap auto: w=%d h=%d aspect=%.3f gap=%.4fin",
                width,
                height,
                aspect,
                cbar_pad_inches,
            )
        else:
            base_pad_inches = float(cbar_pad) * fig_w_in
            cbar_pad_inches = np.clip(
                base_pad_inches, spacing.gap.min_val, spacing.gap.max_val
            )
            logger.debug(
                "single-panel vertical gap override: w=%d h=%d aspect=%.3f raw=%.4fin clipped=%.4fin",
                width,
                height,
                aspect,
                base_pad_inches,
                cbar_pad_inches,
            )
        cbar_pad = cbar_pad_inches / max(fig_w_in, EPSILON_SMALL)

        # --- Right margin (slightly smaller than left) with physical safeguard ---
        right_margin = max(
            outer_margin * spacing.right_margin_scale,
            outer_margin - spacing.right_margin_offset,
        )
        right_margin_inches = right_margin * fig_w_in
        if right_margin_inches < spacing.edge_guard_in:
            right_margin = spacing.edge_guard_in / max(fig_w_in, EPSILON_SMALL)

        # === 2) Compute plot width from panel-fraction rule ===
        available = 1.0 - (
            outer_margin + extra_left_frac + cbar_pad + right_margin + extra_right_frac
        )
        plot_width_candidate = max(
            layout.min_plot_fraction, available / (1.0 + cw_panel)
        )
        aspect_width_target = usable_height
        plot_width = min(plot_width_candidate, aspect_width_target)

        # initial colourbar width (fraction of figure)
        cax_width = cw_panel * plot_width

        # === 3) CAP COLOURBAR PHYSICAL WIDTH ===
        # Compute maximum allowed fraction based on desired physical width
        phys_frac = layout.colourbar.desired_physical_width_in / max(
            fig_w_in, EPSILON_SMALL
        )
        max_cax_frac = min(layout.colourbar.max_fraction_of_fig, phys_frac)

        if cax_width > max_cax_frac:
            cax_width = max_cax_frac
            plot_width = min(
                aspect_width_target,
                max(
                    layout.min_plot_fraction,
                    1.0 - (outer_margin + cbar_pad + cax_width + right_margin),
                ),
            )

        # === 4) Place axes ===
        plot_left = outer_margin + extra_left_frac
        ax = fig.add_axes((plot_left, bottom_val, plot_width, usable_height))

        cax_left = plot_left + plot_width + cbar_pad
        cax = fig.add_axes((cax_left, bottom_val, cax_width, usable_height))

    else:
        cbar_height = (
            layout.colourbar.default_height_frac if cbar_height is None else cbar_height
        )
        cbar_pad = layout.colourbar.default_pad_frac if cbar_pad is None else cbar_pad

        plot_left = outer_margin
        plot_width = 1.0 - 2 * outer_margin
        plot_height = usable_height - (cbar_height + cbar_pad)
        plot_height = max(plot_height, layout.min_plot_fraction)

        ax_bottom = bottom_val + cbar_height + cbar_pad
        ax = fig.add_axes((plot_left, ax_bottom, plot_width, plot_height))
        cax = fig.add_axes((plot_left, bottom_val, plot_width, cbar_height))

    _style_axes([ax])
    _set_axes_limits([ax], width=width, height=height)
    return fig, ax, cax


def build_layout(
    *,
    plot_spec: PlotSpec,
    height: int | None = None,
    width: int | None = None,
    layout_config: LayoutConfig | None = None,
) -> tuple[Figure, list[Axes], dict[str, Axes | None]]:
    """Create a GridSpec layout for multi-panel plots.

    This function can accommodate 2-3 panels with associated colourbars (vertical or horizontal),
    proper spacing, and aspect-ratio-aware sizing.

    Layout Structure:
        2-panel: [Ground Truth] [Prediction] [Colourbar]
        3-panel: [Ground Truth] [Prediction] [Colourbar] [Difference] [Colourbar]

    All spacing parameters are expressed as fractions of figure dimensions to ensure
    consistent scaling across different figure sizes and data aspect ratios.

    Args:
        plot_spec: PlotSpec object containing whether to include a difference panel,
                   colourbar orientation ('vertical'/'horizontal'),
                   colourbar strategy ('shared'/'separate'), and other formatting preferences.
        height: Optional data height for aspect-ratio-aware figure sizing. If provided with width,
           the figure dimensions will be calculated to maintain proper data aspect ratios.
        width: Optional data width for aspect-ratio-aware figure sizing.
        layout_config: Optional LayoutConfig instance to override default layout parameters.
                      If None, uses default configuration.

    Returns:
        tuple containing:
            - Figure: The matplotlib Figure object with GridSpec layout applied
            - list[Axes]: List of main plot axes [ground_truth, prediction, difference*]
                         (*difference only present if plot_spec.include_difference=True)
            - dict[str, Axes | None]: Dictionary with dedicated colourbar axes:
                {"prediction": Axes | None, "difference": Axes | None}

    Raises:
        InvalidArrayError: If the arrays are not 2D or have different shapes.

    """
    layout = layout_config or _DEFAULT_LAYOUT_CONFIG
    outer_margin = layout.outer_margin
    title_space = layout.title_footer.title_space
    footer_space = layout.title_footer.footer_space
    cbar_width = layout.colourbar.default_width_frac
    cbar_height = layout.colourbar.default_height_frac
    cbar_pad = layout.colourbar.default_pad_frac

    # Decide how many main panels are required and which orientation the colourbars use
    n_panels = 3 if plot_spec.include_difference else 2
    orientation = plot_spec.colourbar_location

    # Choose gutter default per orientation
    gutter = (
        layout.gutter_horizontal
        if orientation == "horizontal"
        else layout.gutter_vertical
    )

    # Calculate top boundary: ensure title space does not consume too much of the figure.
    # At least 60% of the figure height is reserved for the plotting area.
    top_val = max(layout.min_usable_height_fraction, 1.0 - (outer_margin + title_space))
    # Calculate bottom boundary, reserving footer space for metadata
    bottom_val = outer_margin + footer_space

    # Calculate figure size based on data aspect ratio or use defaults
    if height and width and height > 0:
        # Calculate panel width maintaining data aspect ratio
        base_h = layout.base_height_in  # Standard height in inches
        aspect = width / height

        if orientation == "vertical":
            panel_w = base_h * aspect
            # Account for colourbars: panels + gutters + colourbar slots
            fig_w = (
                n_panels * panel_w
                + (n_panels - 1) * gutter * panel_w
                + (n_panels - 1) * cbar_width * panel_w
            )
        else:
            # --- Horizontal colourbars ---
            # Portion of height available to the plot row once title and margins are considered
            usable_h_frac = top_val - outer_margin
            plot_row_frac = 1.0 / (1.0 + cbar_pad + cbar_height)

            effective_plot_h = base_h * usable_h_frac * plot_row_frac
            panel_w = effective_plot_h * aspect

            fig_w = n_panels * panel_w + (n_panels - 1) * gutter * panel_w
        fig_size = (fig_w, base_h)
    else:
        # Use predefined sizes when data dimensions are unknown
        fig_size = layout.default_figsizes[n_panels]

    fig = plt.figure(figsize=fig_size, constrained_layout=False, facecolor="none")

    if orientation == "vertical":
        # Delegate to the vertical builder which organises columns for panels and colourbars
        axs, caxes = _build_grid_vertical(
            fig,
            n_panels=n_panels,
            plot_spec=plot_spec,
            outer_margin=outer_margin,
            gutter=gutter,
            cbar_width=cbar_width,
            top_val=top_val,
            bottom_val=bottom_val,
        )
    else:
        # Delegate to the horizontal builder which organises rows for plots and colourbars
        axs, caxes = _build_grid_horizontal(
            fig,
            n_panels=n_panels,
            plot_spec=plot_spec,
            outer_margin=outer_margin,
            gutter=gutter,
            cbar_height=cbar_height,
            cbar_pad=cbar_pad,
            top_val=top_val,
            bottom_val=bottom_val,
        )

    _set_titles(axs, plot_spec)
    _style_axes(axs)

    return fig, axs, caxes


def _build_grid_vertical(  # noqa: PLR0913, C901, PLR0912
    fig: Figure,
    *,
    n_panels: int,
    plot_spec: PlotSpec,
    outer_margin: float,
    gutter: float,
    cbar_width: float,
    top_val: float,
    bottom_val: float,
) -> tuple[list[Axes], dict[str, Axes | None]]:
    """Construct a one-row GridSpec with vertical colourbars.

    Layout overview (left to right):
    - When using shared colourbars: [GroundTruth][Prediction][cbar][gutter][Difference][cbar]
    - When using separate colourbars: [GT][cbar][gutter][Pred][cbar][gutter][Diff][cbar]

    Args:
        fig: The target Matplotlib figure to attach the GridSpec to.
        n_panels: Number of main panels to create (2 or 3).
        plot_spec: Plot configuration containing colourbar strategy and titles.
        outer_margin: Fractional margin applied around the figure.
        gutter: Fractional spacing between panel groups.
        cbar_width: Fractional width allocated to colourbar slots.
        top_val: The top boundary of the usable plotting area (accounts for title space).
        bottom_val: The bottom boundary of the usable plotting area (accounts for
            reserved footer space).

    Returns:
        A tuple of (axes, colourbar_axes) where axes are the main plot axes in order
        and colourbar_axes is a dict containing dedicated colourbar axes if present.

    """
    # Panel arrangement: [GT][Pred][cbar][gutter][Diff][cbar]
    col_specs: list[float] = []
    separate_colourbars = plot_spec.colourbar_strategy == "separate"

    if separate_colourbars:
        # Each panel gets its own colourbar: [GT][cbar][gutter][Pred][cbar][gutter][Diff][cbar]
        for ii in range(n_panels):
            col_specs.append(1.0)
            col_specs.append(cbar_width)
            if ii < n_panels - 1:
                col_specs.append(gutter)
    else:
        for ii in range(n_panels):
            col_specs.append(1.0)
            if ii == PREDICTION_PANEL_INDEX:
                col_specs.append(cbar_width)
            if ii == DIFFERENCE_PANEL_INDEX:
                col_specs.append(cbar_width)
            if ii != n_panels - 1 and ii > 0:
                col_specs.append(gutter)

    # One-row grid; width ratios encode panels, colourbars and gutters
    gs = GridSpec(
        nrows=1,
        ncols=len(col_specs),
        figure=fig,
        width_ratios=col_specs,
        left=outer_margin,
        right=1 - outer_margin,
        top=top_val,
        bottom=bottom_val,
        wspace=0.0,
    )

    axs: list[Axes] = []
    caxes: dict[str, Axes | None] = {
        "groundtruth": None,
        "prediction": None,
        "difference": None,
    }

    col_idx = 0
    for ii in range(n_panels):
        # Create the main panel axis
        ax = fig.add_subplot(gs[0, col_idx])
        axs.append(ax)
        col_idx += 1

        if separate_colourbars:
            panel_names = ["groundtruth", "prediction", "difference"]
            if ii < len(panel_names):
                # Create a dedicated colourbar axis immediately to the right of the panel
                caxes[panel_names[ii]] = fig.add_subplot(gs[0, col_idx])
            col_idx += 1
            if ii < n_panels - 1:
                col_idx += 1
        else:
            if ii == PREDICTION_PANEL_INDEX:
                caxes["prediction"] = fig.add_subplot(gs[0, col_idx])
                col_idx += 1
            if ii == DIFFERENCE_PANEL_INDEX:
                caxes["difference"] = fig.add_subplot(gs[0, col_idx])
                col_idx += 1
            if ii == PREDICTION_PANEL_INDEX and ii != n_panels - 1:
                col_idx += 1

    return axs, caxes


def _build_grid_horizontal(  # noqa: PLR0913, PLR0912
    fig: Figure,
    *,
    n_panels: int,
    plot_spec: PlotSpec,
    outer_margin: float,
    gutter: float,
    cbar_height: float,
    cbar_pad: float,
    top_val: float,
    bottom_val: float,
) -> tuple[list[Axes], dict[str, Axes | None]]:
    """Construct a three-row GridSpec with horizontal colourbars.

    Grid structure (rows):
    - Row 0: Main plot panels
    - Row 1: Padding row to separate plots from colourbars
    - Row 2: Colourbar row (inset axes centred within parent slots)

    Column pattern alternates panels and gutters, with the trailing gutter removed.

    Args:
        fig: The target Matplotlib figure to attach the GridSpec to.
        n_panels: Number of main panels to create (2 or 3).
        plot_spec: Plot configuration containing colourbar strategy.
        outer_margin: Fractional margin applied around the figure.
        gutter: Fractional spacing between panel groups.
        cbar_height: Fractional height allocated to the colourbar row.
        cbar_pad: Fractional padding between the plot row and colourbar row.
        top_val: The top boundary of the usable plotting area (accounts for title space).
        bottom_val: The bottom boundary of the usable plotting area (accounts for
            reserved footer space).

    Returns:
        A tuple of (axes, colourbar_axes) where axes are the main plot axes in order
        and colourbar_axes is a dict containing dedicated colourbar axes if present.

    """
    # Grid structure: 3 rows ([plots][pad][colourbars])
    width_ratios = [1.0, gutter] * n_panels
    width_ratios = width_ratios[:-1]

    # Three-row grid; height ratios encode plot, pad, and colourbar rows
    gs = GridSpec(
        nrows=3,
        ncols=len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[1.0, cbar_pad, cbar_height],
        left=outer_margin,
        right=1 - outer_margin,
        top=top_val,
        bottom=bottom_val,
        wspace=0.0,
        hspace=0.0,
    )

    axs: list[Axes] = []
    separate_colourbars = plot_spec.colourbar_strategy == "separate"
    caxes: dict[str, Axes | None] = {
        "groundtruth": None,
        "prediction": None,
        "difference": None,
    }

    for i in range(n_panels):
        # Columns are [panel, gutter, panel, gutter, ...]; select even columns for panels
        panel_col = 2 * i
        axs.append(fig.add_subplot(gs[0, panel_col]))

    for ax in axs:
        # Anchor to the bottom of the cell to avoid vertical slack above panels
        ax.set_anchor("S")

    def _inset_cbar(parent_ax: Axes) -> Axes:
        """Create a centred inset colourbar axis within parent, using configured fractions."""
        parent_ax.set_axis_off()
        return parent_ax.inset_axes((HCBAR_LEFT_FRAC, 0.0, HCBAR_WIDTH_FRAC, 1.0))

    if separate_colourbars:
        # ---- Separate colourbar logic ----
        # Ground truth bar under first panel (if present)
        if n_panels >= 1:
            caxes["groundtruth"] = _inset_cbar(fig.add_subplot(gs[2, 0]))
        else:
            caxes["groundtruth"] = None

        # Prediction bar under second panel (if present)
        if n_panels >= MIN_PANEL_COUNT_FOR_PREDICTION:
            caxes["prediction"] = _inset_cbar(fig.add_subplot(gs[2, 2]))
        else:
            caxes["prediction"] = None

        # Difference bar under third panel (if present)
        if n_panels >= MIN_PANEL_COUNT_FOR_DIFFERENCE:
            caxes["difference"] = _inset_cbar(fig.add_subplot(gs[2, 4]))
        else:
            caxes["difference"] = None
    else:
        # ---- Shared colourbar logic ----
        # Under GT+gutter+Pred span columns 0..3 when two panels present
        if n_panels >= MIN_PANEL_COUNT_FOR_PREDICTION:
            caxes["prediction"] = _inset_cbar(fig.add_subplot(gs[2, 0:3]))
        else:
            # Fallback: only GT present; use full column
            caxes["prediction"] = _inset_cbar(fig.add_subplot(gs[2, 0:1]))

        # Difference bar under third panel (if present)
        if n_panels >= MIN_PANEL_COUNT_FOR_DIFFERENCE:
            caxes["difference"] = _inset_cbar(fig.add_subplot(gs[2, 4]))
        else:
            caxes["difference"] = None

    return axs, caxes


# --- Helper Functions ---


def _set_titles(axs: list[Axes], plot_spec: PlotSpec) -> None:
    """Set titles for each plot panel based on the PlotSpec configuration.

    The titles are applied in order: ground truth, prediction, and optionally difference.
    For difference panels, the difference mode (e.g., "signed", "absolute") is appended
    to provide clear indication of the type of comparison being shown.

    Args:
        axs: List of matplotlib Axes objects to title (in order: GT, pred, diff)
        plot_spec: PlotSpec containing title strings, whether to include a difference panel,
                   and difference mode configuration.

    """
    # If difference panel is included, append the difference mode to the title
    title_difference = (
        plot_spec.title_difference + f" ({plot_spec.diff_mode})"
        if plot_spec.include_difference
        else None
    )

    titles = [plot_spec.title_groundtruth, plot_spec.title_prediction, title_difference]
    for ax, title in zip(axs, titles, strict=False):
        if title is not None:
            ax.set_title(
                title,
                fontfamily="monospace",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "pad": _DEFAULT_LAYOUT_CONFIG.title_footer.bbox_pad_title,
                    "alpha": 1.0,
                },
            )


def _style_axes(axs: Sequence[Axes]) -> None:
    """Apply consistent styling to all plot axes.

    This function sets up the visual appearance:
    - Removes axis ticks and labels (common for image/map data)
    - Sets equal aspect ratio to prevent distortion of spatial data

    Note: This function only applies styling and does not affect layout.

    Args:
        axs: Sequence of matplotlib Axes objects to style

    """
    for ax in axs:
        ax.axis("off")  # Remove axis ticks, labels, and spines
        ax.set_aspect("equal")  # Maintain square pixels for spatial data


def _set_axes_limits(axs: list[Axes], *, width: int, height: int) -> None:
    """Set consistent axis limits across all plot panels.

    Ensures all axes display the same spatial extent following polar mapping conventions.

    Args:
        axs: List of matplotlib Axes objects to configure
        width: Width of the data array (sets x-axis limits: 0 to width)
        height: Height of the data array (sets y-axis limits: height to 0 for polar data)

    Note:
        The y-axis is inverted (height to 0) to follow environmental science conventions
        for polar data visualisation, where higher latitude values are
        positioned at the top of the display for both Arctic and Antarctic regions.

    """
    for ax in axs:
        ax.set_xlim(0, width)  # X-axis: left edge to right edge
        ax.set_ylim(
            height, 0
        )  # Y-axis: top edge to bottom edge (geographical convention)


# --- Colourbar Functions ---
def get_cbar_limits_from_mappable(cbar: Colorbar) -> tuple[float, float]:
    """Return (vmin, vmax) for a colourbar's mappable with robust fallbacks."""
    vmin = vmax = None
    try:  # Works for many matplotlib mappables
        vmin, vmax = cbar.mappable.get_clim()  # type: ignore[attr-defined]
    except AttributeError:
        norm = getattr(cbar.mappable, "norm", None)
        vmin = getattr(norm, "vmin", None)
        vmax = getattr(norm, "vmax", None)
    if vmin is None or vmax is None:
        vmin, vmax = (
            _DEFAULT_LAYOUT_CONFIG.formatting.default_vmin_fallback,
            _DEFAULT_LAYOUT_CONFIG.formatting.default_vmax_fallback,
        )
    return float(vmin), float(vmax)


def _add_colourbars(  # noqa: PLR0913, PLR0912
    axs: list[Axes],
    *,
    image_groundtruth: QuadContourSet,
    image_prediction: QuadContourSet | None = None,
    image_difference: QuadContourSet | None = None,
    plot_spec: PlotSpec,
    diff_colour_scale: DiffColourmapSpec | None = None,
    display_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
    cbar_axes: dict[str, Axes | None] | None = None,
) -> None:
    """Create and position colourbars.

    This function creates two types of colourbars:
    1. Shared colourbar for ground truth and prediction (same data scale)
    2. Separate colourbar for difference data (different scale, often symmetric)

    The function works with the layout from build_layout, using dedicated
    colourbar axes when available, or falling back to automatic matplotlib placement.

    Colourbar Design:
        - Ground truth/prediction: Uses data range from plot_spec
        - Difference: Often symmetric around zero with specialised tick formatting
        - Orientation: Matches plot_spec.colourbar_location ('vertical'/'horizontal')

    Args:
        axs: List of main plot axes in order [ground_truth, prediction, difference*]
             (*difference only present if include_difference=True)
        image_groundtruth: QuadContourSet from ground truth plot - provides colour mapping
                          and normalisation for shared ground truth/prediction colourbar
        image_prediction: Optional QuadContourSet from prediction plot.
        image_difference: Optional QuadContourSet from difference plot - provides
                         specialised colour mapping (often symmetric) for difference colourbar
        plot_spec: PlotSpec containing colourbar orientation, value ranges, and formatting
        diff_colour_scale: Optional difference colour scale specification.
        display_ranges: Optional tuple of (groundtruth_range, prediction_range) for colourbar limits.
        cbar_axes: Optional dict with pre-allocated colourbar axes from layout builders:
                  {"groundtruth": Axes|None, "prediction": Axes|None, "difference": Axes|None}
                  None values mean "no dedicated axis" and will fall back to automatic placement.

    Layout:
        - cbar_axes["prediction"]: Used for shared ground truth/prediction colourbar
        - cbar_axes["difference"]: Used for difference panel colourbar

    """
    orientation = plot_spec.colourbar_location
    is_vertical = orientation == "vertical"
    separate_colourbars = plot_spec.colourbar_strategy == "separate"

    # Expand display_ranges tuple for clarity
    groundtruth_vmin = display_ranges[0][0] if display_ranges else None
    groundtruth_vmax = display_ranges[0][1] if display_ranges else None
    prediction_vmin = display_ranges[1][0] if display_ranges else None
    prediction_vmax = display_ranges[1][1] if display_ranges else None

    if separate_colourbars:
        # Create individual colourbars for each panel

        # Ground truth colourbar
        if cbar_axes and cbar_axes.get("groundtruth"):
            colourbar_groundtruth = plt.colorbar(
                image_groundtruth, cax=cbar_axes["groundtruth"], orientation=orientation
            )
            format_linear_ticks(
                colourbar_groundtruth,
                vmin=groundtruth_vmin,
                vmax=groundtruth_vmax,
                decimals=1,
                is_vertical=is_vertical,
            )

        # Prediction colourbar
        if image_prediction and cbar_axes and cbar_axes.get("prediction"):
            colourbar_prediction = plt.colorbar(
                image_prediction, cax=cbar_axes["prediction"], orientation=orientation
            )
            format_linear_ticks(
                colourbar_prediction,
                vmin=prediction_vmin,
                vmax=prediction_vmax,
                decimals=1,
                is_vertical=is_vertical,
            )
    else:
        # Create shared colourbar for ground truth and prediction panels
        if cbar_axes is not None and cbar_axes.get("prediction") is not None:
            # Use dedicated colourbar axis (preferred - better layout control)
            colourbar_truth = plt.colorbar(
                image_groundtruth, cax=cbar_axes["prediction"], orientation=orientation
            )
        else:
            # Fallback: automatic positioning across both panels
            colourbar_truth = plt.colorbar(
                image_groundtruth, ax=[axs[0], axs[1]], orientation=orientation
            )

        # Tick formatting
        format_linear_ticks(
            colourbar_truth,
            vmin=groundtruth_vmin,
            vmax=groundtruth_vmax,
            decimals=1,
            is_vertical=is_vertical,
        )

    # Create separate colourbar for difference panel (if present)
    # Difference data often has different scale and symmetric range around zero
    if plot_spec.include_difference and image_difference is not None:
        # Build a fixed ScalarMappable so the colourbar never depends on per-frame QuadContourSet
        sm = None
        if diff_colour_scale is not None:
            if diff_colour_scale.norm is not None:
                sm = plt.cm.ScalarMappable(
                    norm=diff_colour_scale.norm, cmap=diff_colour_scale.cmap
                )
            else:
                sm = plt.cm.ScalarMappable(
                    norm=Normalize(
                        vmin=diff_colour_scale.vmin, vmax=diff_colour_scale.vmax
                    ),
                    cmap=diff_colour_scale.cmap,
                )
            sm.set_array([])

        if cbar_axes is not None and cbar_axes.get("difference") is not None:
            colourbar_diff = plt.colorbar(
                sm if sm is not None else image_difference,
                cax=cbar_axes["difference"],
                orientation=orientation,
            )
        else:
            colourbar_diff = plt.colorbar(
                sm if sm is not None else image_difference,
                ax=axs[2],
                orientation=orientation,
            )

        # Tick formatting: symmetric for TwoSlopeNorm, otherwise linear
        if isinstance(image_difference.norm, TwoSlopeNorm):
            vmin = float(
                image_difference.norm.vmin
                or _DEFAULT_LAYOUT_CONFIG.formatting.default_vmin_diff_fallback
            )
            vmax = float(
                image_difference.norm.vmax
                or _DEFAULT_LAYOUT_CONFIG.formatting.default_vmax_diff_fallback
            )
            format_symmetric_ticks(
                colourbar_diff,
                vmin=vmin,
                vmax=vmax,
                decimals=2,
                is_vertical=is_vertical,
                centre=image_difference.norm.vcenter,
            )
        else:
            format_linear_ticks(colourbar_diff, decimals=2, is_vertical=is_vertical)


# --- Tick Formatting Functions ---


def format_linear_ticks(
    colourbar: Colorbar,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    decimals: int = 1,
    is_vertical: bool,
    use_scientific_notation: bool = False,
) -> None:
    """Format a linear colourbar with 5 ticks.

    If vmin/vmax are not provided, derive them from the colourbar's mappable.

    Args:
        colourbar: Colorbar to format.
        vmin: Minimum value.
        vmax: Maximum value.
        decimals: Number of decimal places for tick labels.
        is_vertical: Whether the colorbar is vertical.
        use_scientific_notation: Whether to format tick labels in scientific notation.

    """
    axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis

    if vmin is None or vmax is None:
        mvmin, mvmax = get_cbar_limits_from_mappable(colourbar)
        vmin = mvmin if vmin is None else vmin
        vmax = mvmax if vmax is None else vmax

    ticks = np.linspace(
        float(vmin), float(vmax), _DEFAULT_LAYOUT_CONFIG.formatting.num_ticks_linear
    )
    colourbar.set_ticks([float(t) for t in ticks])

    if use_scientific_notation:
        axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}e}"))
    else:
        axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}f}"))
    if not is_vertical:
        colourbar.ax.xaxis.set_tick_params(pad=1)
    apply_monospace_to_cbar_text(colourbar)


def format_symmetric_ticks(  # noqa: PLR0913
    colourbar: Colorbar,
    *,
    vmin: float,
    vmax: float,
    decimals: int = 2,
    is_vertical: bool,
    centre: float | None = None,
    use_scientific_notation: bool = False,
) -> None:
    """Format symmetric diverging ticks with a centred midpoint.

    Places five ticks: [vmin, midpoint to centre, centre, centre to midpoint, vmax].

    Args:
        colourbar: Colorbar to format.
        vmin: Minimum value.
        vmax: Maximum value.
        decimals: Number of decimal places for tick labels.
        is_vertical: Whether the colorbar is vertical.
        centre: centre value for diverging colourmap (default: 0.0).
        use_scientific_notation: Whether to format tick labels in scientific notation.

    """
    axis = colourbar.ax.yaxis if is_vertical else colourbar.ax.xaxis
    centre_val = centre if centre is not None else 0.0
    midpoint_factor = _DEFAULT_LAYOUT_CONFIG.formatting.midpoint_factor
    ticks = [
        vmin,
        midpoint_factor * (vmin + centre_val),
        centre_val,
        midpoint_factor * (centre_val + vmax),
        vmax,
    ]
    colourbar.set_ticks([float(t) for t in ticks])

    if use_scientific_notation:
        axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}e}"))
    else:
        axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}f}"))
    if not is_vertical:
        colourbar.ax.xaxis.set_tick_params(pad=1)
    apply_monospace_to_cbar_text(colourbar)


def apply_monospace_to_cbar_text(colourbar: Colorbar) -> None:
    """Set tick labels and axis labels on a colourbar to monospace family."""
    ax = colourbar.ax
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontfamily("monospace")
    # Ensure axis labels also use monospace if present
    ax.xaxis.label.set_fontfamily("monospace")
    ax.yaxis.label.set_fontfamily("monospace")


def _default_vertical_gap_inches(aspect: float, cfg: GapConfig) -> float:
    """Return an aspect-aware physical gap between panel and vertical colourbar."""
    aspect = max(aspect, EPSILON_SMALL)
    if aspect >= 1.0:
        wide = min(aspect, cfg.wide_limit)
        if np.isclose(cfg.wide_limit, 1.0):
            return cfg.base
        t = (wide - 1.0) / (cfg.wide_limit - 1.0)
        gap = cfg.base - t * (cfg.base - cfg.wide_gap)
    else:
        tall_limit = max(cfg.tall_limit, EPSILON_SMALL)
        tall = min(1.0 / aspect, 1.0 / tall_limit)
        denom = (1.0 / tall_limit) - 1.0
        t = 0.0 if np.isclose(denom, 0.0) else (tall - 1.0) / denom
        gap = cfg.base + t * (cfg.tall_gap - cfg.base)
    return float(np.clip(gap, cfg.min_val, cfg.max_val))


# --- Text and Box Annotation Functions ---


def set_suptitle_with_box(fig: Figure, text: str) -> plt.Text:
    """Draw a fixed-position title with a white box that doesn't influence layout.

    Returns the Text artist so callers can update with set_text during animation.
    This version avoids kwargs that are unsupported on older Matplotlib.

    Args:
        fig: Matplotlib Figure object.
        text: Title text to display.

    Returns:
        Text artist for the title.

    """
    config = _DEFAULT_LAYOUT_CONFIG.title_footer
    bbox = {
        "facecolor": "white",
        "edgecolor": "none",
        "pad": config.bbox_pad_title,
        "alpha": 1.0,
    }
    t = fig.text(
        x=0.5,
        y=config.title_y,
        s=text,
        ha="center",
        va="top",
        fontsize=config.title_fontsize,
        fontfamily="monospace",
        transform=fig.transFigure,
        bbox=bbox,
    )
    with contextlib.suppress(Exception):
        t.set_zorder(config.zorder_high)
    return t


def set_footer_with_box(fig: Figure, text: str) -> plt.Text:
    """Draw a fixed-position footer with a white box at bottom centre.

    Footer is intended for metadata and secondary information.

    Args:
        fig: Matplotlib Figure object.
        text: Footer text to display.

    Returns:
        Text artist for the footer.

    """
    config = _DEFAULT_LAYOUT_CONFIG.title_footer
    bbox = {
        "facecolor": "white",
        "edgecolor": "none",
        "pad": config.bbox_pad_title,
        "alpha": 1.0,
    }
    t = fig.text(
        x=0.5,
        y=config.footer_y,
        s=text,
        ha="center",
        va="bottom",
        fontsize=config.footer_fontsize,
        fontfamily="monospace",
        transform=fig.transFigure,
        bbox=bbox,
    )
    with contextlib.suppress(Exception):
        t.set_zorder(config.zorder_high)
    return t


def draw_badge_with_box(fig: Figure, x: float, y: float, text: str) -> plt.Text:
    """Draw a warning/info badge with white background box at figure coords.

    Args:
        fig: Matplotlib Figure object.
        x: X position in figure coordinates (0-1).
        y: Y position in figure coordinates (0-1).
        text: Badge text to display.

    Returns:
        Text artist for the badge.

    """
    config = _DEFAULT_LAYOUT_CONFIG.title_footer
    bbox = {
        "facecolor": "white",
        "edgecolor": "none",
        "pad": config.bbox_pad_badge,
        "alpha": 1.0,
    }
    t = fig.text(
        x=x,
        y=y,
        s=text,
        fontsize=config.footer_fontsize,
        fontfamily="monospace",
        color="firebrick",
        ha="center",
        va="top",
        bbox=bbox,
    )
    with contextlib.suppress(Exception):
        t.set_zorder(config.zorder_high)
    return t
