from dataclasses import replace
from datetime import date, datetime
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib import pyplot as plt

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.helpers import (
    _build_footer_static,
    _build_footer_video,
    _build_title_video,
    _clear_plot,
    _draw_frame,
    _draw_main_panels,
    _draw_warning_badge,
    _format_date_for_title,
    _format_title,
    _formatted_variable_name,
    _maybe_add_footer,
    _prepare_difference,
    _prepare_static_plot,
    _safe_linspace,
)
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.plotting_core import make_diff_colourmap


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


class TestDrawWarningBadge:
    def test_no_warnings_is_a_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Skip drawing entirely when there are no warnings."""
        fake_draw_badge = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.helpers.draw_badge_with_box", fake_draw_badge
        )
        fig = plt.figure()

        _draw_warning_badge(fig, None, [])

        fake_draw_badge.assert_not_called()

    def test_without_title_text_uses_default_y(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fall back to a fixed y-position when no title text object is given."""
        fake_draw_badge = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.helpers.draw_badge_with_box", fake_draw_badge
        )
        fig = plt.figure()

        _draw_warning_badge(fig, None, ["COLOUR ISSUE: bad"])

        fake_draw_badge.assert_called_once()
        assert fake_draw_badge.call_args.args[2] == pytest.approx(0.90)

    def test_with_title_text_positions_below_title(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Position the badge just below a real title's y-coordinate."""
        fake_draw_badge = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.helpers.draw_badge_with_box", fake_draw_badge
        )
        fig = plt.figure()
        title_text = fig.suptitle("Test title", y=0.95)

        _draw_warning_badge(fig, title_text, ["COLOUR ISSUE: bad"])

        fake_draw_badge.assert_called_once()
        assert fake_draw_badge.call_args.args[2] < 0.95


class TestMaybeAddFooter:
    def test_disabled_footer_metadata_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skip drawing when include_footer_metadata is False."""
        fake_set_footer = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.helpers.set_footer_with_box", fake_set_footer
        )
        spec = replace(DEFAULT_SIC_SPEC, include_footer_metadata=False)
        fig = plt.figure()

        _maybe_add_footer(fig, spec)

        fake_set_footer.assert_not_called()

    def test_empty_footer_text_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skip drawing when there is no metadata subtitle to show."""
        fake_set_footer = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.helpers.set_footer_with_box", fake_set_footer
        )
        spec = replace(
            DEFAULT_SIC_SPEC, include_footer_metadata=True, metadata_subtitle=None
        )
        fig = plt.figure()

        _maybe_add_footer(fig, spec)

        fake_set_footer.assert_not_called()

    def test_draws_footer_when_metadata_subtitle_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Draw the footer when a metadata subtitle is present."""
        fake_set_footer = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.helpers.set_footer_with_box", fake_set_footer
        )
        spec = replace(
            DEFAULT_SIC_SPEC,
            include_footer_metadata=True,
            metadata_subtitle="epochs=50",
        )
        fig = plt.figure()

        _maybe_add_footer(fig, spec)

        fake_set_footer.assert_called_once_with(fig, "epochs=50")

    def test_swallows_footer_drawing_errors(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Log and continue if drawing the footer raises."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.helpers.set_footer_with_box",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        spec = replace(
            DEFAULT_SIC_SPEC,
            include_footer_metadata=True,
            metadata_subtitle="epochs=50",
        )
        fig = plt.figure()

        with caplog.at_level("ERROR"):
            _maybe_add_footer(fig, spec)

        assert "Failed to draw footer" in caplog.text


class TestFormatTitle:
    def test_with_hemisphere_and_units(self) -> None:
        """Include hemisphere and units when both are given."""
        result = _format_title("2t", "north", date(2020, 1, 1), "K")

        assert result == "2t [K] (North)   Shown: 2020-01-01"

    def test_without_hemisphere_or_units(self) -> None:
        """Omit hemisphere and units segments when neither is given."""
        result = _format_title("2t", None, date(2020, 1, 1), None)

        assert result == "2t   Shown: 2020-01-01"

    def test_accepts_datetime(self) -> None:
        """Accept a datetime and format only its date portion."""
        result = _format_title("2t", None, datetime(2020, 1, 1, 12, 30), None)

        assert result == "2t   Shown: 2020-01-01"


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
    def test_requires_diff_colour_scale_when_including_difference(self) -> None:
        """Reject a missing colour scale when a difference panel is requested."""
        spec = replace(DEFAULT_SIC_SPEC, include_difference=True)
        fig, axs = plt.subplots(1, 3)
        ground_truth = np.full((4, 4), 0.5, dtype=np.float32)
        prediction = np.full((4, 4), 0.5, dtype=np.float32)

        with pytest.raises(InvalidArrayError, match="diff_colour_scale"):
            _draw_frame(list(axs), ground_truth, prediction, spec, LandMask(None))
        plt.close(fig)

    def test_signed_difference_uses_two_slope_norm(self) -> None:
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
            LandMask(None),
            diff_colour_scale=colour_scale,
        )

        assert image_difference is not None
        plt.close(fig)

    def test_absolute_difference_uses_vmin_vmax(self) -> None:
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
            LandMask(None),
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


class TestFormattedVariableName:
    def test_replaces_underscores_and_title_cases(self) -> None:
        """Turn a snake_case variable name into a human-friendly title."""
        assert (
            _formatted_variable_name("sea_ice_concentration") == "Sea Ice Concentration"
        )

    def test_empty_string_stays_empty(self) -> None:
        """Return an empty string unchanged."""
        assert _formatted_variable_name("") == ""


class TestFormatDateForTitle:
    def test_date_object(self) -> None:
        """Format a plain date object as an ISO date string."""
        assert _format_date_for_title(date(2023, 12, 25)) == "2023-12-25"

    def test_datetime_object_drops_time(self) -> None:
        """Format a datetime object, stripping the time component."""
        assert _format_date_for_title(datetime(2023, 12, 25, 14, 30)) == "2023-12-25"


class TestBuildTitleVideo:
    def test_empty_dates_omits_frame_segment(self) -> None:
        """Omit the 'Frame:' segment entirely when no dates are given."""
        result = _build_title_video("sea_ice_concentration", DEFAULT_SIC_SPEC, [], 0)

        assert "Frame:" not in result
        assert result.endswith("Prediction")


class TestBuildFooterStatic:
    def test_includes_metadata_subtitle_when_present(self) -> None:
        """Include the metadata subtitle line when set."""
        spec = replace(DEFAULT_SIC_SPEC, metadata_subtitle="epochs=50")

        assert _build_footer_static(spec) == "epochs=50"

    def test_empty_when_no_metadata_subtitle(self) -> None:
        """Return an empty string when there is no metadata subtitle."""
        spec = replace(DEFAULT_SIC_SPEC, metadata_subtitle=None)

        assert _build_footer_static(spec) == ""


class TestBuildFooterVideo:
    def test_includes_metadata_subtitle_alongside_animation_range(self) -> None:
        """Include both the animation range and the metadata subtitle."""
        spec = replace(DEFAULT_SIC_SPEC, metadata_subtitle="epochs=50")
        dates: list[Any] = [date(2020, 1, 1), date(2020, 1, 5)]

        result = _build_footer_video(spec, dates)

        assert "Animating from 2020-01-01 to 2020-01-05" in result
        assert "epochs=50" in result

    def test_empty_dates_omits_animation_range(self) -> None:
        """Omit the animation-range line when no dates are given."""
        spec = replace(DEFAULT_SIC_SPEC, metadata_subtitle=None)

        assert _build_footer_video(spec, []) == ""
