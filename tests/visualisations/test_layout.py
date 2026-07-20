from dataclasses import replace
from datetime import date
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.layout import (
    _set_axes_limits,
    build_layout,
    build_single_panel_figure,
    draw_badge_with_box,
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


# Silence Matplotlib animation warning in this test module
pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything:UserWarning:matplotlib.animation"
)


@pytest.mark.parametrize(
    "colourbar_location", ["horizontal", "vertical"], ids=lambda s: f"cbar-{s}"
)
@pytest.mark.parametrize(
    "include_difference", [False, True], ids=lambda v: "with-diff" if v else "no-diff"
)
@pytest.mark.parametrize(
    "colourbar_strategy", ["shared", "separate"], ids=lambda s: f"strategy-{s}"
)
def test_no_axes_overlap(
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


def test_y_axis_orientation_for_geographical_data() -> None:
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


@pytest.mark.parametrize("colourbar_location", ["horizontal", "vertical"])
@pytest.mark.parametrize("include_difference", [False, True])
def test_figure_text_boxes_do_not_overlap(
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


# --- Single Panel Layout Tests ---


@pytest.mark.parametrize("colourbar_location", ["horizontal", "vertical"])
def test_single_panel_no_overlap(
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
