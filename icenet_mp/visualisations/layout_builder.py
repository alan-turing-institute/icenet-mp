"""Figure/GridSpec construction for map comparison layouts.

Handles the layout of the plot panels for maps and colourbars. Used through the
plotting system to create comparison visualisations with 2-3 panels (ground
truth, prediction, optional difference), plus standalone single-panel figures.
"""

import logging
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from icenet_mp.types import PlotSpec

from .layout import EPSILON_SMALL, GapConfig, LayoutConfig

logger = logging.getLogger(__name__)

# Panel index constants (used for layout logic)
PREDICTION_PANEL_INDEX = 1
DIFFERENCE_PANEL_INDEX = 2
MIN_PANEL_COUNT_FOR_PREDICTION = 2
MIN_PANEL_COUNT_FOR_DIFFERENCE = 3

# Horizontal colourbar inset constants
HCBAR_WIDTH_FRAC = 0.75  # centred 75% width
HCBAR_LEFT_FRAC = (1.0 - HCBAR_WIDTH_FRAC) / 2.0


class LayoutBuilder:
    """Builds GridSpec-based figures/axes for single- and multi-panel comparison maps."""

    def build_single_panel_figure(  # noqa: PLR0913, PLR0915
        self,
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

        layout = layout_config or LayoutConfig()
        outer_margin = layout.outer_margin
        title_space = layout.title_footer.title_space
        footer_space = layout.title_footer.footer_space

        base_h = layout.base_height_in
        aspect = width / max(1, height)
        fig_w = base_h * aspect
        fig = plt.figure(
            figsize=(fig_w, base_h), constrained_layout=False, facecolor="none"
        )

        top_val = max(
            layout.min_usable_height_fraction, 1.0 - (outer_margin + title_space)
        )
        bottom_val = outer_margin + footer_space
        usable_height = top_val - bottom_val

        if colourbar_location == "vertical":
            # --- Interpret cbar_width as "fraction of panel width" ---
            cw_panel = float(
                layout.colourbar.default_width_frac
                if cbar_width is None
                else cbar_width
            )
            fig_w_in = float(fig.get_size_inches()[0])
            cw_panel = layout.colourbar.clamp_frac(cw_panel, fig_w_in)

            spacing = layout.single_panel_spacing

            extra_left_frac = spacing.outer_buffer_in / max(fig_w_in, EPSILON_SMALL)
            extra_right_frac = spacing.outer_buffer_in / max(fig_w_in, EPSILON_SMALL)

            if cbar_pad is None:
                cbar_pad_inches = self._default_vertical_gap_inches(aspect, spacing.gap)
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
                outer_margin
                + extra_left_frac
                + cbar_pad
                + right_margin
                + extra_right_frac
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
                layout.colourbar.default_height_frac
                if cbar_height is None
                else cbar_height
            )
            cbar_pad = (
                layout.colourbar.default_pad_frac if cbar_pad is None else cbar_pad
            )

            plot_left = outer_margin
            plot_width = 1.0 - 2 * outer_margin
            plot_height = usable_height - (cbar_height + cbar_pad)
            plot_height = max(plot_height, layout.min_plot_fraction)

            ax_bottom = bottom_val + cbar_height + cbar_pad
            ax = fig.add_axes((plot_left, ax_bottom, plot_width, plot_height))
            cax = fig.add_axes((plot_left, bottom_val, plot_width, cbar_height))

        self._style_axes([ax])
        self.set_axes_limits([ax], width=width, height=height)
        return fig, ax, cax

    def build_layout(
        self,
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
        layout = layout_config or LayoutConfig()
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
        top_val = max(
            layout.min_usable_height_fraction, 1.0 - (outer_margin + title_space)
        )
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
            axs, caxes = self._build_grid_vertical(
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
            axs, caxes = self._build_grid_horizontal(
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

        self.set_titles(axs, plot_spec)
        self._style_axes(axs)

        return fig, axs, caxes

    def _build_grid_vertical(  # noqa: PLR0913, C901, PLR0912
        self,
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
        self,
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

    def set_titles(self, axs: list[Axes], plot_spec: PlotSpec) -> None:
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

        titles = [
            plot_spec.title_groundtruth,
            plot_spec.title_prediction,
            title_difference,
        ]
        for ax, title in zip(axs, titles, strict=False):
            if title is not None:
                ax.set_title(
                    title,
                    fontfamily="monospace",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "pad": LayoutConfig().title_footer.bbox_pad_title,
                        "alpha": 1.0,
                    },
                )

    def _style_axes(self, axs: Sequence[Axes]) -> None:
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

    def set_axes_limits(self, axs: list[Axes], *, width: int, height: int) -> None:
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

    def _default_vertical_gap_inches(self, aspect: float, cfg: GapConfig) -> float:
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
