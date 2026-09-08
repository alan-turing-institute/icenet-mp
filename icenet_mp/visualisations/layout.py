"""Plot Map Layout Configuration and Text Annotation Primitives.

Defines the tunable layout configuration dataclasses shared by LayoutBuilder and
ColourbarFormatter, plus the low-level fixed-position text/box drawing primitives
used for suptitles, footers and warning badges.
"""

import contextlib
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
        fig, axs, cax = LayoutBuilder().build_single_panel_figure(
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
