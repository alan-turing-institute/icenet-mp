from datetime import date, datetime
from typing import Any

from icenet_mp.types import Metadata, PlotSpec
from icenet_mp.visualisations.plot_annotator import PlotAnnotator


class TestFormatTitle:
    def test_with_hemisphere_and_units(self) -> None:
        """Include hemisphere and units when both are given."""
        annotator = PlotAnnotator(PlotSpec(hemisphere="north"))

        result = annotator.format_title("2t", date(2020, 1, 1), "K")

        assert result == "2t [K] (North)   Shown: 2020-01-01"

    def test_without_hemisphere_or_units(self) -> None:
        """Omit hemisphere and units segments when neither is given."""
        annotator = PlotAnnotator(PlotSpec(hemisphere=None))

        result = annotator.format_title("2t", date(2020, 1, 1), None)

        assert result == "2t   Shown: 2020-01-01"

    def test_accepts_datetime(self) -> None:
        """Accept a datetime and format only its date portion."""
        annotator = PlotAnnotator(PlotSpec(hemisphere=None))

        result = annotator.format_title("2t", datetime(2020, 1, 1, 12, 30), None)

        assert result == "2t   Shown: 2020-01-01"


class TestFormattedVariableName:
    def test_replaces_underscores_and_title_cases(self) -> None:
        """Turn a snake_case variable name into a human-friendly title."""
        result = PlotAnnotator(PlotSpec()).formatted_variable_name(
            "sea_ice_concentration"
        )

        assert result == "Sea Ice Concentration"

    def test_empty_string_stays_empty(self) -> None:
        """Return an empty string unchanged."""
        assert PlotAnnotator(PlotSpec()).formatted_variable_name("") == ""


class TestFormatDateForTitle:
    def test_date_object(self) -> None:
        """Format a plain date object as an ISO date string."""
        result = PlotAnnotator(PlotSpec()).format_date_for_title(date(2023, 12, 25))

        assert result == "2023-12-25"

    def test_datetime_object_drops_time(self) -> None:
        """Format a datetime object, stripping the time component."""
        result = PlotAnnotator(PlotSpec()).format_date_for_title(
            datetime(2023, 12, 25, 14, 30)
        )

        assert result == "2023-12-25"


class TestBuildTitleVideo:
    def test_empty_dates_omits_frame_segment(self) -> None:
        """Omit the 'Frame:' segment entirely when no dates are given."""
        result = PlotAnnotator(PlotSpec()).title_for_video(
            "sea_ice_concentration", [], 0
        )

        assert "Frame:" not in result
        assert result.endswith("Prediction")


class TestBuildFooterStatic:
    def test_includes_metadata_subtitle_when_set(self) -> None:
        """Include the metadata subtitle line once metadata has been set."""
        annotator = PlotAnnotator(PlotSpec())
        annotator.set_metadata(Metadata(model="unet"))

        assert annotator.footer_for_static() == "Model: unet"

    def test_empty_when_metadata_never_set(self) -> None:
        """Return an empty string when no metadata has been set."""
        assert PlotAnnotator(PlotSpec()).footer_for_static() == ""

    def test_empty_when_metadata_has_no_facts(self) -> None:
        """Return an empty string when the set metadata formats to nothing."""
        annotator = PlotAnnotator(PlotSpec())
        annotator.set_metadata(Metadata())

        assert annotator.footer_for_static() == ""


class TestBuildFooterVideo:
    def test_includes_metadata_subtitle_alongside_animation_range(self) -> None:
        """Include both the animation range and the metadata subtitle."""
        annotator = PlotAnnotator(PlotSpec())
        annotator.set_metadata(Metadata(model="unet"))
        dates: list[Any] = [date(2020, 1, 1), date(2020, 1, 5)]

        result = annotator.footer_for_video(dates)

        assert "Animating from 2020-01-01 to 2020-01-05" in result
        assert "Model: unet" in result

    def test_empty_dates_omits_animation_range(self) -> None:
        """Omit the animation-range line when no dates are given."""
        assert PlotAnnotator(PlotSpec()).footer_for_video([]) == ""


class TestFormatSubtitle:
    def test_formats_model_epoch_and_training_data(self) -> None:
        """Format model, epoch, dates and training data into a multi-line subtitle."""
        metadata = Metadata(
            model="test_model",
            current_epoch=5,
            start="2020-01-01",
            end="2020-01-10",
            cadence="1d",
            n_points=10,
            vars_by_source={"era5": ["2t", "sp"]},
        )

        subtitle = PlotAnnotator(PlotSpec()).format_subtitle(metadata)

        assert subtitle is not None
        assert "Model: test_model" in subtitle
        assert "Epoch: 5" in subtitle
        assert "Training Data:" in subtitle
        assert "2020-01-01" in subtitle
        assert "2020-01-10" in subtitle
        assert "10 pts" in subtitle

    def test_includes_history_window(self) -> None:
        """Mention the history window when n_history_steps is set."""
        metadata = Metadata(
            start="2020-01-01",
            end="2020-01-10",
            cadence="1d",
            n_history_steps=3,
        )

        subtitle = PlotAnnotator(PlotSpec()).format_subtitle(metadata)

        assert subtitle is not None
        assert "3 step history" in subtitle

    def test_lists_source_with_no_variables(self) -> None:
        """List a source with an empty variable list without parentheses."""
        metadata = Metadata(vars_by_source={"era5": []})

        subtitle = PlotAnnotator(PlotSpec()).format_subtitle(metadata)

        assert subtitle is not None
        assert "Training Data: era5" in subtitle
        assert "era5 (" not in subtitle

    def test_minimal_metadata_returns_none(self) -> None:
        """Return None when no metadata fields are set."""
        assert PlotAnnotator(PlotSpec()).format_subtitle(Metadata()) is None
