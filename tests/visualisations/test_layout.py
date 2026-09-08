from dataclasses import replace
from datetime import date
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.text import Text

from icenet_mp.types import DiffColourmapSpec
from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.layout import (
    GapConfig,
    LayoutConfig,
    _add_colourbars,
    _build_grid_horizontal,
    _default_vertical_gap_inches,
    _set_axes_limits,
    build_layout,
    build_single_panel_figure,
    draw_badge_with_box,
    format_symmetric_ticks,
    get_cbar_limits_from_mappable,
    set_footer_with_box,
    set_suptitle_with_box,
)

EPSILON: float = 1e-6
RECTANGLE = tuple[float, float, float, float]  # (Left, Bottom, Right, Top)


def axis_rectangle(ax: Axes) -> RECTANGLE:
    """Return (left, bottom, right, top) in figure-normalised coords [0, 1]."""
    bbox = ax.get_position()
    return (bbox.x0, bbox.y0, bbox.x1, bbox.y1)


def rectangles_overlap(
    rect_a: RECTANGLE,
    rect_b: RECTANGLE,
    *,
    epsilon: float = EPSILON,
) -> bool:
    """True if two axis-aligned rectangles overlap (with tolerance)."""
    la, ba, ra, ta = rect_a
    lb, bb, rb, tb = rect_b
    separated_h = ra <= lb + epsilon or rb <= la + epsilon
    separated_v = ta <= bb + epsilon or tb <= ba + epsilon
    return not (separated_h or separated_v)


def _text_rectangle(fig: Figure, text_artist: Text) -> RECTANGLE:
    """Return text bounding box in figure-normalised coords [0, 1]."""
    fig.canvas.draw()
    # Obtain a renderer in a backend-agnostic, mypy-friendly way
    canvas: Any = fig.canvas
    get_renderer = getattr(canvas, "get_renderer", None)
    if callable(get_renderer):
        renderer = get_renderer()
    else:
        renderer = getattr(canvas, "renderer", None)
    bbox = text_artist.get_window_extent(renderer=renderer)
    # Transform display to figure coords
    (x0, y0), (x1, y1) = fig.transFigure.inverted().transform(
        [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)]
    )
    return (float(x0), float(y0), float(x1), float(y1))


# Silence Matplotlib animation warning in this test module
pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything:UserWarning:matplotlib.animation"
)


class TestBuildLayoutOverlap:
    """Panels, colourbars and figure text must never overlap for common configurations."""

    @pytest.mark.parametrize(
        "colourbar_location", ["horizontal", "vertical"], ids=lambda s: f"cbar-{s}"
    )
    @pytest.mark.parametrize(
        "include_difference",
        [False, True],
        ids=lambda v: "with-diff" if v else "no-diff",
    )
    @pytest.mark.parametrize(
        "colourbar_strategy", ["shared", "separate"], ids=lambda s: f"strategy-{s}"
    )
    def test_no_axes_overlap(
        self,
        sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
        *,
        colourbar_location: str,
        include_difference: bool,
        colourbar_strategy: str,
    ) -> None:
        """Panels and colourbars must not overlap for common layout configurations."""
        ground_truth, _, _ = sic_pair_2d

        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_location=colourbar_location,  # type: ignore[arg-type]
            include_difference=include_difference,
            colourbar_strategy=colourbar_strategy,  # type: ignore[arg-type]
        )

        _, axes, colourbar_axes = build_layout(
            plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
        )

        # Collect rectangles for all visible axes: main panels + any colourbars that exist.
        rectangles = [axis_rectangle(ax) for ax in axes]
        rectangles.extend(
            axis_rectangle(ax) for ax in colourbar_axes.values() if ax is not None
        )

        # Pairwise non-overlap
        for rect_a, rect_b in combinations(rectangles, 2):
            assert not rectangles_overlap(rect_a, rect_b), (
                f"Found overlap between rectangles {rect_a} and {rect_b}"
            )

    @pytest.mark.parametrize("colourbar_location", ["horizontal", "vertical"])
    @pytest.mark.parametrize("include_difference", [False, True])
    @pytest.mark.parametrize("colourbar_strategy", ["shared", "separate"])
    def test_axes_have_reasonable_gaps(
        self,
        sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
        *,
        colourbar_location: str,
        include_difference: bool,
        colourbar_strategy: str,
    ) -> None:
        """Require a small horizontal gutter between side-by-side axes."""
        ground_truth, _, _ = sic_pair_2d

        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_location=colourbar_location,  # type: ignore[arg-type]
            include_difference=include_difference,
            colourbar_strategy=colourbar_strategy,  # type: ignore[arg-type]
        )

        _, axes, colourbar_axes = build_layout(
            plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
        )

        rectangles = [axis_rectangle(ax) for ax in axes]
        rectangles.extend(
            axis_rectangle(ax) for ax in colourbar_axes.values() if ax is not None
        )

        def _min_horizontal_gap(a: RECTANGLE, b: RECTANGLE) -> float:
            la, _, ra, _ = a
            lb, _, rb, _ = b
            if ra <= lb:  # a left of b
                return lb - ra
            if rb <= la:  # b left of a
                return la - rb
            return 0.0  # not horizontally ordered

        for rect_a, rect_b in combinations(rectangles, 2):
            # Skip overlapping pairs (covered by test_no_axes_overlap)
            if rectangles_overlap(rect_a, rect_b):
                continue

            # Only enforce horizontal spacing (rows are allowed to touch vertically)
            horizontal_gap = _min_horizontal_gap(rect_a, rect_b)
            if horizontal_gap > 0.0:
                assert horizontal_gap >= 0.005, (
                    f"Expected ≥0.5% horizontal gap, got {horizontal_gap:.5f} for {rect_a} vs {rect_b}"
                )

    @pytest.mark.parametrize("colourbar_location", ["horizontal", "vertical"])
    @pytest.mark.parametrize("include_difference", [False, True])
    def test_figure_text_boxes_do_not_overlap(
        self,
        sic_pair_2d: tuple[np.ndarray, np.ndarray, date],
        *,
        colourbar_location: str,
        include_difference: bool,
    ) -> None:
        """Ensure figure title, warning badge, and footer do not overlap panels or colourbars."""
        ground_truth, _, _ = sic_pair_2d

        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_location=colourbar_location,  # type: ignore[arg-type]
            include_difference=include_difference,
        )

        fig, axes, caxes = build_layout(
            plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
        )

        # Add figure-level title, warning badge (synthetic), and footer
        title = set_suptitle_with_box(fig, "Title")
        ty = title.get_position()[1]
        badge = draw_badge_with_box(fig, 0.5, max(ty - 0.05, 0.0), "Warnings: example")
        footer = set_footer_with_box(fig, "Footer metadata")

        # Collect rectangles: panels, colourbar axes, and figure texts
        rectangles = [axis_rectangle(ax) for ax in axes]
        rectangles.extend(axis_rectangle(ax) for ax in caxes.values() if ax is not None)
        rectangles.append(_text_rectangle(fig, title))
        rectangles.append(_text_rectangle(fig, badge))
        rectangles.append(_text_rectangle(fig, footer))

        # No overlaps among any of these elements
        for rect_a, rect_b in combinations(rectangles, 2):
            assert not rectangles_overlap(rect_a, rect_b), (
                f"Found overlap between rectangles {rect_a} and {rect_b}"
            )


class TestSetAxesLimits:
    def test_y_axis_orientation_for_geographical_data(self) -> None:
        """Test that y-axis is oriented correctly for geographical data following polar mapping conventions."""
        # Create a simple figure with axes
        fig, ax = plt.subplots()

        # Test with sample dimensions
        height, width = 48, 64

        # Apply the axis limits function
        _set_axes_limits([ax], width=width, height=height)

        # Check that y-axis is inverted for geographical convention
        # For polar data (both Arctic and Antarctic), higher latitude values should be
        # positioned at the top of the display, following environmental science conventions
        y_min, y_max = ax.get_ylim()

        # Check if the axis is properly oriented for geographical data
        # We want higher latitude values (pole-ward) at the top of the display
        if y_max > y_min:
            # Normal orientation: y_max at top
            assert y_max == height, f"Y-axis maximum should be {height}, got {y_max}"
            assert y_min == 0, f"Y-axis minimum should be 0 , got {y_min}"
        else:
            # Inverted orientation: y_min at top
            assert y_min == height, f"Y-axis minimum should be {height}, got {y_min}"
            assert y_max == 0, f"Y-axis maximum should be 0 , got {y_max}"

        # Check x-axis is still normal (left to right)
        x_min, x_max = ax.get_xlim()
        assert x_min == 0, f"X-axis minimum should be 0, got {x_min}"
        assert x_max == width, f"X-axis maximum should be {width}, got {x_max}"

        plt.close(fig)


class TestBuildSinglePanelFigure:
    @pytest.mark.parametrize("colourbar_location", ["horizontal", "vertical"])
    def test_single_panel_no_overlap(
        self,
        era5_temperature_2d: np.ndarray,
        *,
        colourbar_location: str,
    ) -> None:
        """Single panel layout: main panel and colourbar must not overlap."""
        height, width = era5_temperature_2d.shape

        fig, ax, cax = build_single_panel_figure(
            height=height,
            width=width,
            colourbar_location=colourbar_location,  # type: ignore[arg-type]
        )

        # Get rectangles for panel and colourbar
        panel_rect = axis_rectangle(ax)
        cbar_rect = axis_rectangle(cax)

        # They should not overlap
        assert not rectangles_overlap(panel_rect, cbar_rect), (
            f"Panel and colourbar overlap: panel={panel_rect}, cbar={cbar_rect}"
        )

        plt.close(fig)

    @pytest.mark.parametrize("colourbar_location", ["horizontal", "vertical"])
    def test_single_panel_has_reasonable_gap(
        self,
        era5_temperature_2d: np.ndarray,
        *,
        colourbar_location: str,
    ) -> None:
        """Single panel layout: require a minimum gap between panel and colourbar."""
        height, width = era5_temperature_2d.shape

        fig, ax, cax = build_single_panel_figure(
            height=height,
            width=width,
            colourbar_location=colourbar_location,  # type: ignore[arg-type]
        )

        panel_rect = axis_rectangle(ax)
        cbar_rect = axis_rectangle(cax)

        _, pb, pr, _ = panel_rect
        cl, _, _, ct = cbar_rect

        if colourbar_location == "vertical":
            # Vertical colorbar should be to the right of panel
            gap = cl - pr
            assert gap >= 0.005, (
                f"Expected ≥0.5% gap between panel and vertical colorbar, got {gap:.5f}"
            )
        else:  # horizontal
            # Horizontal colorbar should be below panel
            gap = pb - ct
            assert gap >= 0.005, (
                f"Expected ≥0.5% gap between panel and horizontal colorbar, got {gap:.5f}"
            )

        plt.close(fig)

    def test_single_panel_with_text_annotations(
        self,
        era5_temperature_2d: np.ndarray,
    ) -> None:
        """Single panel: title and annotations should not overlap panel or colourbar."""
        height, width = era5_temperature_2d.shape

        fig, ax, cax = build_single_panel_figure(
            height=height,
            width=width,
            colourbar_location="vertical",
        )

        # Add text annotations
        title = set_suptitle_with_box(fig, "Test Title")
        footer = set_footer_with_box(fig, "Test Footer")

        # Collect all rectangles
        panel_rect = axis_rectangle(ax)
        cbar_rect = axis_rectangle(cax)
        title_rect = _text_rectangle(fig, title)
        footer_rect = _text_rectangle(fig, footer)

        rectangles = [panel_rect, cbar_rect, title_rect, footer_rect]

        # No overlaps
        for rect_a, rect_b in combinations(rectangles, 2):
            assert not rectangles_overlap(rect_a, rect_b), (
                f"Found overlap between {rect_a} and {rect_b}"
            )

        plt.close(fig)

    @pytest.mark.parametrize(
        ("height", "width"),
        [
            (48, 48),  # Square
            (181, 720),  # Wide (ERA5-like)
            (432, 432),  # Square (OSISAF-like)
            (100, 200),  # Wide
            (200, 100),  # Tall
        ],
    )
    def test_single_panel_various_aspect_ratios(
        self,
        height: int,
        width: int,
    ) -> None:
        """Single panel layout should handle various aspect ratios without overlap."""
        fig, ax, cax = build_single_panel_figure(
            height=height,
            width=width,
            colourbar_location="vertical",
        )

        panel_rect = axis_rectangle(ax)
        cbar_rect = axis_rectangle(cax)

        # No overlap
        assert not rectangles_overlap(panel_rect, cbar_rect), (
            f"Overlap for {height}x{width}: panel={panel_rect}, cbar={cbar_rect}"
        )

        plt.close(fig)


# --- Validation and Edge-Case Tests ---


class TestBuildSinglePanelFigureValidation:
    """build_single_panel_figure must reject non-positive data dimensions."""

    def test_raises_for_non_positive_height(self) -> None:
        with pytest.raises(ValueError, match="height and width must be positive"):
            build_single_panel_figure(height=0, width=10, colourbar_location="vertical")

    def test_raises_for_non_positive_width(self) -> None:
        with pytest.raises(ValueError, match="height and width must be positive"):
            build_single_panel_figure(
                height=10, width=-1, colourbar_location="vertical"
            )


class TestSinglePanelVerticalGapOverride:
    """Explicit cbar_pad takes the override branch and is clipped to GapConfig bounds."""

    def test_large_cbar_pad_is_clamped_to_max(self) -> None:
        fig, ax, cax = build_single_panel_figure(
            height=100, width=100, colourbar_location="vertical", cbar_pad=100.0
        )
        fig_w_in = fig.get_size_inches()[0]
        expected_gap = LayoutConfig().single_panel_spacing.gap.max_val / fig_w_in

        _, _, ax_right, _ = axis_rectangle(ax)
        cax_left, _, _, _ = axis_rectangle(cax)
        assert (cax_left - ax_right) == pytest.approx(expected_gap, abs=1e-4)

        plt.close(fig)

    def test_small_cbar_pad_is_clamped_to_min(self) -> None:
        fig, ax, cax = build_single_panel_figure(
            height=100, width=100, colourbar_location="vertical", cbar_pad=1e-8
        )
        fig_w_in = fig.get_size_inches()[0]
        expected_gap = LayoutConfig().single_panel_spacing.gap.min_val / fig_w_in

        _, _, ax_right, _ = axis_rectangle(ax)
        cax_left, _, _, _ = axis_rectangle(cax)
        assert (cax_left - ax_right) == pytest.approx(expected_gap, abs=1e-4)

        plt.close(fig)


class TestSinglePanelColourbarWidthClamp:
    """A very wide panel forces the physical-width clamp on the colourbar."""

    def test_extreme_wide_aspect_clamps_colourbar_to_physical_max(self) -> None:
        # aspect=100 forces the minimum-width floor above the physical-width limit,
        # which in turn forces cax_width above max_cax_frac and triggers the clamp.
        fig, ax, cax = build_single_panel_figure(
            height=20, width=2000, colourbar_location="vertical"
        )
        fig_w_in = fig.get_size_inches()[0]
        expected_cax_width = (
            LayoutConfig().colourbar.desired_physical_width_in / fig_w_in
        )

        cbar_rect = axis_rectangle(cax)
        observed_cax_width = cbar_rect[2] - cbar_rect[0]
        assert observed_cax_width == pytest.approx(expected_cax_width, rel=1e-3)

        # Sanity: panel and colourbar still do not overlap after the clamp.
        assert not rectangles_overlap(axis_rectangle(ax), cbar_rect)

        plt.close(fig)


class TestBuildLayoutDefaultFigsize:
    """When data dimensions are unknown, build_layout falls back to default_figsizes."""

    def test_falls_back_for_three_panels(self) -> None:
        spec = replace(DEFAULT_SIC_SPEC, include_difference=True)
        fig, _, _ = build_layout(plot_spec=spec)
        assert tuple(fig.get_size_inches()) == LayoutConfig().default_figsizes[3]
        plt.close(fig)

    def test_falls_back_for_two_panels(self) -> None:
        spec = replace(DEFAULT_SIC_SPEC, include_difference=False)
        fig, _, _ = build_layout(plot_spec=spec)
        assert tuple(fig.get_size_inches()) == LayoutConfig().default_figsizes[2]
        plt.close(fig)


class TestBuildGridHorizontalSinglePanel:
    """_build_grid_horizontal with n_panels=1 only occurs when called directly.

    build_layout always derives n_panels as 2 or 3, so the "only ground truth present"
    fallback branches inside _build_grid_horizontal can only be reached by calling the
    private helper directly, matching this file's existing precedent of importing
    private helpers (e.g. _set_axes_limits) for targeted coverage.
    """

    def test_separate_strategy_with_single_panel_has_only_groundtruth_colourbar(
        self,
    ) -> None:
        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_strategy="separate",
            colourbar_location="horizontal",
        )
        fig = plt.figure(figsize=(6, 6))
        axs, caxes = _build_grid_horizontal(
            fig,
            n_panels=1,
            plot_spec=spec,
            outer_margin=0.05,
            gutter=0.03,
            cbar_height=0.07,
            cbar_pad=0.03,
            top_val=0.85,
            bottom_val=0.13,
        )
        assert len(axs) == 1
        assert caxes["groundtruth"] is not None
        assert caxes["prediction"] is None
        assert caxes["difference"] is None
        plt.close(fig)

    def test_shared_strategy_with_single_panel_falls_back_to_full_column(self) -> None:
        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_strategy="shared",
            colourbar_location="horizontal",
        )
        fig = plt.figure(figsize=(6, 6))
        axs, caxes = _build_grid_horizontal(
            fig,
            n_panels=1,
            plot_spec=spec,
            outer_margin=0.05,
            gutter=0.03,
            cbar_height=0.07,
            cbar_pad=0.03,
            top_val=0.85,
            bottom_val=0.13,
        )
        assert len(axs) == 1
        assert caxes["prediction"] is not None
        assert caxes["difference"] is None
        plt.close(fig)


class TestGetCbarLimitsFromMappable:
    """get_cbar_limits_from_mappable falls back through mappable.norm, then defaults."""

    def test_falls_back_to_norm_attributes_without_get_clim(
        self, monkeypatch: pytest.MonkeyPatch, era5_temperature_2d: np.ndarray
    ) -> None:
        fig, ax, cax = build_single_panel_figure(
            height=16, width=16, colourbar_location="vertical"
        )
        image = ax.contourf(era5_temperature_2d, levels=10)
        cbar = plt.colorbar(image, cax=cax)

        class _FakeNorm:
            vmin = 260.0
            vmax = 290.0

        class _FakeMappable:
            norm = _FakeNorm()

        # _FakeMappable deliberately has no get_clim, forcing the AttributeError fallback.
        monkeypatch.setattr(cbar, "mappable", _FakeMappable())

        vmin, vmax = get_cbar_limits_from_mappable(cbar)
        assert vmin == pytest.approx(260.0)
        assert vmax == pytest.approx(290.0)

        plt.close(fig)

    def test_falls_back_to_default_when_norm_has_no_limits(
        self, monkeypatch: pytest.MonkeyPatch, era5_temperature_2d: np.ndarray
    ) -> None:
        fig, ax, cax = build_single_panel_figure(
            height=16, width=16, colourbar_location="vertical"
        )
        image = ax.contourf(era5_temperature_2d, levels=10)
        cbar = plt.colorbar(image, cax=cax)

        class _FakeMappable:
            norm = None

        monkeypatch.setattr(cbar, "mappable", _FakeMappable())

        vmin, vmax = get_cbar_limits_from_mappable(cbar)
        assert vmin == pytest.approx(0.0)
        assert vmax == pytest.approx(1.0)

        plt.close(fig)


class TestAddColourbars:
    """_add_colourbars: dedicated cbar_axes vs. automatic-placement fallback."""

    def test_separate_strategy_uses_dedicated_axes_for_each_panel(
        self, sic_pair_2d: tuple[np.ndarray, np.ndarray, date]
    ) -> None:
        ground_truth, prediction, _ = sic_pair_2d
        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_strategy="separate",
            colourbar_location="vertical",
            include_difference=False,
        )
        fig, axs, cbar_axes = build_layout(
            plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
        )
        image_groundtruth = axs[0].contourf(ground_truth, levels=10)
        image_prediction = axs[1].contourf(prediction, levels=10)

        _add_colourbars(
            axs,
            image_groundtruth=image_groundtruth,
            image_prediction=image_prediction,
            plot_spec=spec,
            cbar_axes=cbar_axes,
        )

        assert cbar_axes["groundtruth"] is not None
        assert cbar_axes["prediction"] is not None
        assert len(cbar_axes["groundtruth"].get_yticks()) == 5
        assert len(cbar_axes["prediction"].get_yticks()) == 5

        plt.close(fig)

    def test_shared_strategy_falls_back_to_automatic_placement_without_cbar_axes(
        self, sic_pair_2d: tuple[np.ndarray, np.ndarray, date]
    ) -> None:
        ground_truth, prediction, _ = sic_pair_2d
        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_strategy="shared",
            colourbar_location="vertical",
            include_difference=False,
        )
        fig, axs, _ = build_layout(
            plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
        )
        image_groundtruth = axs[0].contourf(ground_truth, levels=10)
        image_prediction = axs[1].contourf(prediction, levels=10)

        n_axes_before = len(fig.axes)
        _add_colourbars(
            axs,
            image_groundtruth=image_groundtruth,
            image_prediction=image_prediction,
            plot_spec=spec,
            cbar_axes=None,
        )
        # The fallback path creates one new automatically-placed colourbar axis.
        assert len(fig.axes) == n_axes_before + 1

        plt.close(fig)


class TestAddColourbarsDifferencePanel:
    """_add_colourbars difference-panel branches: signed (TwoSlopeNorm) vs absolute."""

    def test_signed_difference_uses_symmetric_tick_formatting(
        self, sic_pair_2d: tuple[np.ndarray, np.ndarray, date]
    ) -> None:
        ground_truth, prediction, _ = sic_pair_2d
        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_strategy="shared",
            colourbar_location="vertical",
            include_difference=True,
            diff_mode="signed",
        )
        fig, axs, cbar_axes = build_layout(
            plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
        )
        image_groundtruth = axs[0].contourf(ground_truth, levels=10)
        image_prediction = axs[1].contourf(prediction, levels=10)

        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        image_difference = axs[2].contourf(
            prediction - ground_truth, levels=10, cmap="RdBu_r", norm=norm
        )
        diff_colour_scale = DiffColourmapSpec(
            norm=norm, vmin=None, vmax=None, cmap="RdBu_r"
        )

        _add_colourbars(
            axs,
            image_groundtruth=image_groundtruth,
            image_prediction=image_prediction,
            image_difference=image_difference,
            plot_spec=spec,
            diff_colour_scale=diff_colour_scale,
            cbar_axes=cbar_axes,
        )

        diff_cax = cbar_axes["difference"]
        assert diff_cax is not None
        ticks = diff_cax.get_yticks()
        assert len(ticks) == 5
        # Symmetric ticks: [vmin, mid, centre, mid, vmax], centre defaults to 0.0.
        assert ticks[2] == pytest.approx(0.0, abs=1e-6)
        assert ticks[0] == pytest.approx(-ticks[-1])

        plt.close(fig)

    def test_absolute_difference_without_cbar_axes_uses_automatic_placement(
        self, sic_pair_2d: tuple[np.ndarray, np.ndarray, date]
    ) -> None:
        ground_truth, prediction, _ = sic_pair_2d
        spec = replace(
            DEFAULT_SIC_SPEC,
            colourbar_strategy="shared",
            colourbar_location="vertical",
            include_difference=True,
            diff_mode="absolute",
        )
        fig, axs, _ = build_layout(
            plot_spec=spec, height=ground_truth.shape[0], width=ground_truth.shape[1]
        )
        image_groundtruth = axs[0].contourf(ground_truth, levels=10)
        image_prediction = axs[1].contourf(prediction, levels=10)
        image_difference = axs[2].contourf(
            np.abs(prediction - ground_truth), levels=10, cmap="magma"
        )
        # norm=None routes through the plain Normalize(vmin, vmax) construction branch.
        diff_colour_scale = DiffColourmapSpec(
            norm=None, vmin=0.0, vmax=1.0, cmap="magma"
        )

        n_axes_before = len(fig.axes)
        _add_colourbars(
            axs,
            image_groundtruth=image_groundtruth,
            image_prediction=image_prediction,
            image_difference=image_difference,
            plot_spec=spec,
            diff_colour_scale=diff_colour_scale,
            cbar_axes=None,
        )
        # Both the shared GT/prediction fallback and the difference fallback fire,
        # each creating one new automatically-placed colourbar axis.
        assert len(fig.axes) == n_axes_before + 2

        plt.close(fig)


class TestFormatSymmetricTicksScientificNotation:
    """format_symmetric_ticks supports scientific-notation tick labels."""

    def test_use_scientific_notation_sets_exponential_formatter(self) -> None:
        fig, ax, cax = build_single_panel_figure(
            height=16, width=16, colourbar_location="vertical"
        )
        image = ax.contourf(np.random.default_rng(1).random((16, 16)), levels=10)
        cbar = plt.colorbar(image, cax=cax)

        format_symmetric_ticks(
            cbar, vmin=-1.0, vmax=1.0, is_vertical=True, use_scientific_notation=True
        )

        formatter = cbar.ax.yaxis.get_major_formatter()
        formatted = formatter(0.123456, 0)
        assert "e" in formatted.lower()

        plt.close(fig)


class TestDefaultVerticalGapInchesWideLimit:
    """_default_vertical_gap_inches early-returns cfg.base when wide_limit is ~1.0."""

    def test_wide_limit_equal_to_one_returns_base_gap(self) -> None:
        cfg = GapConfig(wide_limit=1.0)
        gap = _default_vertical_gap_inches(2.0, cfg)
        assert gap == pytest.approx(cfg.base)
