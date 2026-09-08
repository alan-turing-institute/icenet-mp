from dataclasses import replace

import numpy as np
import pytest
from matplotlib import pyplot as plt

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.difference_calculator import DifferenceCalculator
from icenet_mp.visualisations.helpers import (
    _clear_plot,
    _draw_frame,
    _draw_main_panels,
    _prepare_difference,
    _prepare_static_plot,
    _safe_linspace,
)
from icenet_mp.visualisations.land_mask import LandMask

make_diff_colourmap = DifferenceCalculator().make_diff_colourmap


class TestSafeLinspace:
    def test_normal_range(self) -> None:
        """Return an increasing linspace for a normal, finite range."""
        result = _safe_linspace(0.0, 1.0, 5)

        np.testing.assert_allclose(result, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_non_finite_inputs_fall_back_to_unit_interval(self) -> None:
        """Fall back to [0, 1] when either bound is non-finite."""
        result = _safe_linspace(np.nan, 5.0, 3)

        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_swapped_bounds_are_reordered(self) -> None:
        """Swap vmin/vmax when passed in the wrong order."""
        result = _safe_linspace(5.0, 1.0, 3)

        np.testing.assert_allclose(result, [1.0, 3.0, 5.0])

    def test_equal_bounds_produce_a_tiny_range(self) -> None:
        """Produce a tiny non-degenerate range when vmin == vmax."""
        result = _safe_linspace(2.0, 2.0, 3)

        assert result[0] == pytest.approx(2.0)
        assert result[-1] > 2.0


class TestPrepareStaticPlot:
    def test_rejects_mismatched_shapes(self) -> None:
        """Reject ground truth and prediction arrays with different shapes."""
        ground_truth = np.zeros((4, 4), dtype=np.float32)
        prediction = np.zeros((4, 5), dtype=np.float32)

        with pytest.raises(InvalidArrayError, match="different shape"):
            _prepare_static_plot(DEFAULT_SIC_SPEC, ground_truth, prediction)

    def test_no_warnings_returns_no_layout_config(self) -> None:
        """Return a None layout config and no warnings for well-behaved data."""
        ground_truth = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
        prediction = ground_truth.copy()

        height, width, layout_config, warnings = _prepare_static_plot(
            DEFAULT_SIC_SPEC, ground_truth, prediction
        )

        assert (height, width) == (4, 4)
        assert layout_config is None
        assert warnings == []

    def test_warnings_return_a_layout_config_with_extra_title_space(self) -> None:
        """Reserve extra title space when the range-check report has warnings."""
        ground_truth = np.zeros((8, 8), dtype=np.float32)
        prediction = np.full((8, 8), 5.0, dtype=np.float32)

        _height, _width, layout_config, warnings = _prepare_static_plot(
            DEFAULT_SIC_SPEC, ground_truth, prediction
        )

        assert warnings
        assert layout_config is not None
        assert layout_config.title_footer.title_space == pytest.approx(0.10)


class TestPrepareDifference:
    def test_include_difference_false_returns_none(self) -> None:
        """Skip difference computation entirely when disabled."""
        spec = replace(DEFAULT_SIC_SPEC, include_difference=False)
        ground_truth = np.zeros((4, 4), dtype=np.float32)
        prediction = np.ones((4, 4), dtype=np.float32)

        difference, colour_scale = _prepare_difference(spec, ground_truth, prediction)

        assert difference is None
        assert colour_scale is None

    def test_include_difference_true_computes_both(self) -> None:
        """Compute a difference array and matching colour scale when enabled."""
        spec = replace(DEFAULT_SIC_SPEC, include_difference=True, diff_mode="signed")
        ground_truth = np.full((4, 4), 0.8, dtype=np.float32)
        prediction = np.full((4, 4), 0.2, dtype=np.float32)

        difference, colour_scale = _prepare_difference(spec, ground_truth, prediction)

        assert difference is not None
        np.testing.assert_allclose(difference, 0.6, atol=1e-6)
        assert colour_scale is not None


class TestDrawMainPanels:
    def test_levels_override_is_used_for_both_panels(self) -> None:
        """Use the given levels directly when levels_override is provided."""
        fig, axs = plt.subplots(1, 2)
        ground_truth = np.full((4, 4), 0.5, dtype=np.float32)
        prediction = np.full((4, 4), 0.5, dtype=np.float32)
        levels = np.linspace(0.0, 1.0, 11)

        image_gt, image_pred = _draw_main_panels(
            list(axs),
            ground_truth,
            prediction,
            DEFAULT_SIC_SPEC,
            ((0.0, 1.0), (0.0, 1.0)),
            levels_override=levels,
        )

        np.testing.assert_allclose(image_gt.levels, levels)
        np.testing.assert_allclose(image_pred.levels, levels)
        plt.close(fig)

    def test_separate_strategy_uses_independent_levels(self) -> None:
        """Compute independent contour levels per panel under the separate strategy."""
        spec = replace(DEFAULT_SIC_SPEC, colourbar_strategy="separate")
        fig, axs = plt.subplots(1, 2)
        ground_truth = np.full((4, 4), 0.5, dtype=np.float32)
        prediction = np.full((4, 4), 0.5, dtype=np.float32)

        image_gt, image_pred = _draw_main_panels(
            list(axs),
            ground_truth,
            prediction,
            spec,
            ((0.0, 1.0), (2.0, 3.0)),
        )

        assert image_gt.levels[0] == pytest.approx(0.0)
        assert image_gt.levels[-1] == pytest.approx(1.0)
        assert image_pred.levels[0] == pytest.approx(2.0)
        assert image_pred.levels[-1] == pytest.approx(3.0)
        plt.close(fig)


class TestDrawFrame:
    def test_requires_diff_colour_scale_when_including_difference(
        self, no_land_mask: LandMask
    ) -> None:
        """Reject a missing colour scale when a difference panel is requested."""
        spec = replace(DEFAULT_SIC_SPEC, include_difference=True)
        fig, axs = plt.subplots(1, 3)
        ground_truth = np.full((4, 4), 0.5, dtype=np.float32)
        prediction = np.full((4, 4), 0.5, dtype=np.float32)

        with pytest.raises(InvalidArrayError, match="diff_colour_scale"):
            _draw_frame(list(axs), ground_truth, prediction, spec, no_land_mask)
        plt.close(fig)

    def test_signed_difference_uses_two_slope_norm(
        self, no_land_mask: LandMask
    ) -> None:
        """Draw the difference panel using the provided TwoSlopeNorm for signed diffs."""
        spec = replace(DEFAULT_SIC_SPEC, include_difference=True, diff_mode="signed")
        fig, axs = plt.subplots(1, 3)
        ground_truth = np.full((4, 4), 0.8, dtype=np.float32)
        prediction = np.full((4, 4), 0.2, dtype=np.float32)
        difference = ground_truth - prediction
        colour_scale = make_diff_colourmap(difference, mode="signed")

        _, _, image_difference, _ = _draw_frame(
            list(axs),
            ground_truth,
            prediction,
            spec,
            no_land_mask,
            diff_colour_scale=colour_scale,
        )

        assert image_difference is not None
        plt.close(fig)

    def test_absolute_difference_uses_vmin_vmax(self, no_land_mask: LandMask) -> None:
        """Draw the difference panel using explicit vmin/vmax for absolute diffs."""
        spec = replace(DEFAULT_SIC_SPEC, include_difference=True, diff_mode="absolute")
        fig, axs = plt.subplots(1, 3)
        ground_truth = np.full((4, 4), 0.8, dtype=np.float32)
        prediction = np.full((4, 4), 0.2, dtype=np.float32)
        difference = np.abs(ground_truth - prediction)
        colour_scale = make_diff_colourmap(difference, mode="absolute")

        _, _, image_difference, _ = _draw_frame(
            list(axs),
            ground_truth,
            prediction,
            spec,
            no_land_mask,
            diff_colour_scale=colour_scale,
        )

        assert image_difference is not None
        assert colour_scale.norm is None
        plt.close(fig)


class TestClearPlot:
    def test_removes_title_and_contour_collections(self) -> None:
        """Reset the axes title and drop any contour collections."""
        fig, ax = plt.subplots()
        ax.set_title("Some title")
        ax.contourf(np.zeros((4, 4)))
        assert ax.get_title() == "Some title"

        _clear_plot(ax)

        assert ax.get_title() == ""
        assert len(ax.collections) == 0
        plt.close(fig)
