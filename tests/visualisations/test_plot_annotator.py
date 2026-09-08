from dataclasses import replace
from datetime import date, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from matplotlib import pyplot as plt

from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.plot_annotator import PlotAnnotator


class TestDrawWarningBadge:
    def test_no_warnings_is_a_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Skip drawing entirely when there are no warnings."""
        fake_draw_badge = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.plot_annotator.draw_badge_with_box",
            fake_draw_badge,
        )
        fig = plt.figure()

        PlotAnnotator().warning_badge(fig, None, [])

        fake_draw_badge.assert_not_called()

    def test_without_title_text_uses_default_y(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fall back to a fixed y-position when no title text object is given."""
        fake_draw_badge = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.plot_annotator.draw_badge_with_box",
            fake_draw_badge,
        )
        fig = plt.figure()

        PlotAnnotator().warning_badge(fig, None, ["COLOUR ISSUE: bad"])

        fake_draw_badge.assert_called_once()
        assert fake_draw_badge.call_args.args[2] == pytest.approx(0.90)

    def test_with_title_text_positions_below_title(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Position the badge just below a real title's y-coordinate."""
        fake_draw_badge = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.plot_annotator.draw_badge_with_box",
            fake_draw_badge,
        )
        fig = plt.figure()
        title_text = fig.suptitle("Test title", y=0.95)

        PlotAnnotator().warning_badge(fig, title_text, ["COLOUR ISSUE: bad"])

        fake_draw_badge.assert_called_once()
        assert fake_draw_badge.call_args.args[2] < 0.95


class TestMaybeAddFooter:
    def test_disabled_footer_metadata_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skip drawing when include_footer_metadata is False."""
        fake_set_footer = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.plot_annotator.set_footer_with_box",
            fake_set_footer,
        )
        spec = replace(DEFAULT_SIC_SPEC, include_footer_metadata=False)
        fig = plt.figure()

        PlotAnnotator().maybe_add_footer(fig, spec)

        fake_set_footer.assert_not_called()

    def test_empty_footer_text_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skip drawing when there is no metadata subtitle to show."""
        fake_set_footer = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.plot_annotator.set_footer_with_box",
            fake_set_footer,
        )
        spec = replace(
            DEFAULT_SIC_SPEC, include_footer_metadata=True, metadata_subtitle=None
        )
        fig = plt.figure()

        PlotAnnotator().maybe_add_footer(fig, spec)

        fake_set_footer.assert_not_called()

    def test_draws_footer_when_metadata_subtitle_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Draw the footer when a metadata subtitle is present."""
        fake_set_footer = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.visualisations.plot_annotator.set_footer_with_box",
            fake_set_footer,
        )
        spec = replace(
            DEFAULT_SIC_SPEC,
            include_footer_metadata=True,
            metadata_subtitle="epochs=50",
        )
        fig = plt.figure()

        PlotAnnotator().maybe_add_footer(fig, spec)

        fake_set_footer.assert_called_once_with(fig, "epochs=50")

    def test_swallows_footer_drawing_errors(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Log and continue if drawing the footer raises."""
        monkeypatch.setattr(
            "icenet_mp.visualisations.plot_annotator.set_footer_with_box",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        spec = replace(
            DEFAULT_SIC_SPEC,
            include_footer_metadata=True,
            metadata_subtitle="epochs=50",
        )
        fig = plt.figure()

        with caplog.at_level("ERROR"):
            PlotAnnotator().maybe_add_footer(fig, spec)

        assert "Failed to draw footer" in caplog.text


class TestFormatTitle:
    def test_with_hemisphere_and_units(self) -> None:
        """Include hemisphere and units when both are given."""
        result = PlotAnnotator().format_title("2t", "north", date(2020, 1, 1), "K")

        assert result == "2t [K] (North)   Shown: 2020-01-01"

    def test_without_hemisphere_or_units(self) -> None:
        """Omit hemisphere and units segments when neither is given."""
        result = PlotAnnotator().format_title("2t", None, date(2020, 1, 1), None)

        assert result == "2t   Shown: 2020-01-01"

    def test_accepts_datetime(self) -> None:
        """Accept a datetime and format only its date portion."""
        result = PlotAnnotator().format_title(
            "2t", None, datetime(2020, 1, 1, 12, 30), None
        )

        assert result == "2t   Shown: 2020-01-01"


class TestFormattedVariableName:
    def test_replaces_underscores_and_title_cases(self) -> None:
        """Turn a snake_case variable name into a human-friendly title."""
        result = PlotAnnotator().formatted_variable_name("sea_ice_concentration")

        assert result == "Sea Ice Concentration"

    def test_empty_string_stays_empty(self) -> None:
        """Return an empty string unchanged."""
        assert PlotAnnotator().formatted_variable_name("") == ""


class TestFormatDateForTitle:
    def test_date_object(self) -> None:
        """Format a plain date object as an ISO date string."""
        result = PlotAnnotator().format_date_for_title(date(2023, 12, 25))

        assert result == "2023-12-25"

    def test_datetime_object_drops_time(self) -> None:
        """Format a datetime object, stripping the time component."""
        result = PlotAnnotator().format_date_for_title(datetime(2023, 12, 25, 14, 30))

        assert result == "2023-12-25"


class TestBuildTitleVideo:
    def test_empty_dates_omits_frame_segment(self) -> None:
        """Omit the 'Frame:' segment entirely when no dates are given."""
        result = PlotAnnotator().title_for_video(
            "sea_ice_concentration", DEFAULT_SIC_SPEC, [], 0
        )

        assert "Frame:" not in result
        assert result.endswith("Prediction")


class TestBuildFooterStatic:
    def test_includes_metadata_subtitle_when_present(self) -> None:
        """Include the metadata subtitle line when set."""
        spec = replace(DEFAULT_SIC_SPEC, metadata_subtitle="epochs=50")

        assert PlotAnnotator().footer_for_static(spec) == "epochs=50"

    def test_empty_when_no_metadata_subtitle(self) -> None:
        """Return an empty string when there is no metadata subtitle."""
        spec = replace(DEFAULT_SIC_SPEC, metadata_subtitle=None)

        assert PlotAnnotator().footer_for_static(spec) == ""


class TestBuildFooterVideo:
    def test_includes_metadata_subtitle_alongside_animation_range(self) -> None:
        """Include both the animation range and the metadata subtitle."""
        spec = replace(DEFAULT_SIC_SPEC, metadata_subtitle="epochs=50")
        dates: list[Any] = [date(2020, 1, 1), date(2020, 1, 5)]

        result = PlotAnnotator().footer_for_video(spec, dates)

        assert "Animating from 2020-01-01 to 2020-01-05" in result
        assert "epochs=50" in result

    def test_empty_dates_omits_animation_range(self) -> None:
        """Omit the animation-range line when no dates are given."""
        spec = replace(DEFAULT_SIC_SPEC, metadata_subtitle=None)

        assert PlotAnnotator().footer_for_video(spec, []) == ""
