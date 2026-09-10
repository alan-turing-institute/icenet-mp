from typing import Any

import pytest
from omegaconf import DictConfig

from icenet_mp.callbacks.plotting_callback import PlottingCallback
from icenet_mp.types import Metadata
from icenet_mp.visualisations.metadata import (
    build_metadata,
    calculate_training_points,
    extract_cadence_from_config,
    extract_training_date_range,
    extract_variables_by_source,
    format_cadence_display,
    format_metadata_subtitle,
)


@pytest.mark.parametrize(
    ("start", "end", "freq", "expected"),
    [
        # Daily cadence tests
        ("2020-01-01", "2020-01-10", "1d", 10),  # 10 days inclusive
        ("2020-01-01", "2020-01-10", "2d", 5),  # 10 days / 2 = 5 points
        ("2020-01-01", "2020-01-01", "1d", 1),  # Single day
        ("2020-01-01", "2020-01-02", "1d", 2),  # Two days inclusive
        (
            "2020-01-01",
            "2020-01-05",
            "3d",
            1,
        ),  # 5 days / 3 = 1 point (Jan 1), Jan 4 is beyond range
        # Hourly cadence tests
        ("2020-01-01", "2020-01-01", "1h", 24),  # Single day = 24 hours
        ("2020-01-01", "2020-01-01", "24h", 1),  # 24h = daily
        ("2020-01-01T00:00:00", "2020-01-01T23:00:00", "1h", 24),  # With time component
        ("2020-01-01", "2020-01-02", "12h", 4),  # 2 days * 24h / 12h = 4 points
        ("2020-01-01", "2020-01-01", "3h", 8),  # 24h / 3h = 8 points
        ("2020-01-01", "2020-01-01", "2hour", 12),  # "...hour" suffix: 24h / 2 = 12
        ("2020-01-01", "2020-01-01", "2hr", 12),  # "...hr" (not "hour") suffix
        # Format variations
        ("2020-01-01", "2020-01-10", "daily", 10),  # Word format
        ("2020-01-01", "2020-01-01", "hourly", 24),
    ],
)
def test_calculate_training_points_parametric(
    start: str, end: str, freq: str, expected: int
) -> None:
    """Test calculate_training_points with various date ranges and cadences."""
    result = calculate_training_points(start, end, freq)
    assert result == expected, (
        f"Expected {expected} points for {start} to {end} with {freq} cadence"
    )


def test_calculate_training_points_invalid_returns_none() -> None:
    """Test that invalid inputs return None."""
    assert calculate_training_points(None, "2020-01-01", "1d") is None
    assert calculate_training_points("2020-01-01", None, "1d") is None
    assert calculate_training_points("2020-01-01", "2020-01-10", None) is None
    assert calculate_training_points("", "2020-01-01", "1d") is None
    # Note: "invalid" might parse as "1d" due to substring matching - this is acceptable
    # The function is permissive with formats, so truly invalid formats that fail
    # completely should return None
    assert calculate_training_points("2020-01-01", "2020-01-10", "0d") is None
    assert calculate_training_points("2020-01-01", "2020-01-10", "-1h") is None
    # These should return None due to unrecognised format
    assert calculate_training_points("2020-01-01", "2020-01-10", "xyz") is None
    assert calculate_training_points("2020-01-01", "2020-01-10", "123") is None


def test_calculate_training_points_malformed_date_returns_none() -> None:
    """Test that a date string _inclusive_days can't parse returns None, not raise."""
    assert calculate_training_points("not-a-date", "2020-01-10", "1d") is None


def test_format_cadence_display() -> None:
    """Test cadence display formatting."""
    assert format_cadence_display("1d") == "daily"
    assert format_cadence_display("1day") == "daily"
    assert format_cadence_display("1 day") == "daily"
    assert format_cadence_display("1h") == "hourly"
    assert format_cadence_display("1hr") == "hourly"
    assert format_cadence_display("3d") == "3d"  # Not 1d, so unchanged
    assert format_cadence_display("24h") == "24h"  # Not 1h, so unchanged
    assert format_cadence_display(None) is None


def test_extract_variables_by_source_sic() -> None:
    """Test extraction of sea ice variables from dataset config."""
    config = {
        "data": {
            "datasets": {
                "sic1": {
                    "name": "osisaf-sicsouth",
                    "group_as": "osisaf-south",
                },
                "sic2": {
                    "name": "osisaf-sicnorth",
                    "group_as": "osisaf-north",
                },
            },
        },
    }
    result = extract_variables_by_source(config)
    assert result == {
        "osisaf-south": ["sea ice"],
        "osisaf-north": ["sea ice"],
    }


def test_extract_variables_by_source_weather() -> None:
    """Test extraction of weather variables from dataset config."""
    config = {
        "data": {
            "datasets": {
                "era5_1": {
                    "name": "era5-weather",
                    "group_as": "era5",
                    "input": {
                        "join": [
                            {"mars": {"param": ["2t", "sp"]}},
                            {"mars": {"param": ["10u", "10v"]}},
                        ]
                    },
                },
            },
        },
    }
    result = extract_variables_by_source(config)
    # Should extract and sort params
    assert result["era5"] == ["10u", "10v", "2t", "sp"]


def test_extract_variables_by_source_weather_fallback() -> None:
    """Test weather dataset with no params returns empty (no fallback)."""
    config = {
        "data": {
            "datasets": {
                "era5_1": {
                    "name": "era5-weather",
                    "group_as": "era5",
                    "input": {},
                },
            },
        }
    }
    result = extract_variables_by_source(config)
    # When no params are found, the dataset is skipped (no fallback to "weather")
    assert "era5" not in result or result["era5"] == []


def test_extract_variables_by_source_empty_config() -> None:
    """Test with empty or invalid config."""
    assert extract_variables_by_source({}) == {}
    assert extract_variables_by_source({"datasets": {}}) == {}
    assert extract_variables_by_source({"datasets": None}) == {}


def test_extract_variables_by_source_datasets_not_a_dict() -> None:
    """Test that a non-dict 'datasets' value returns an empty result."""
    config: dict[str, Any] = {"data": {"datasets": "not-a-dict"}}
    assert extract_variables_by_source(config) == {}


def test_extract_variables_by_source_skips_non_dict_dataset_entries() -> None:
    """Test that a non-dict dataset entry is skipped rather than raising."""
    config: dict[str, Any] = {
        "data": {
            "datasets": {
                "bad": "not-a-dict",
                "sic1": {"name": "osisaf-sicsouth", "group_as": "osisaf-south"},
            },
        },
    }
    assert extract_variables_by_source(config) == {"osisaf-south": ["sea ice"]}


def test_extract_variables_by_source_skips_missing_group_as() -> None:
    """Test that a dataset without a string 'group_as' is skipped."""
    config: dict[str, Any] = {
        "data": {
            "datasets": {
                "sic1": {"name": "osisaf-sicsouth"},
            },
        },
    }
    assert extract_variables_by_source(config) == {}


def test_extract_variables_by_source_weather_missing_input_key() -> None:
    """Test a weather dataset with no 'input' key at all yields no variables."""
    config: dict[str, Any] = {
        "data": {
            "datasets": {
                "era5_1": {"name": "era5-weather", "group_as": "era5"},
            },
        },
    }
    result = extract_variables_by_source(config)
    assert "era5" not in result


def test_extract_variables_by_source_weather_join_not_a_list() -> None:
    """Test a weather dataset whose 'join' value is not a list yields no variables."""
    config: dict[str, Any] = {
        "data": {
            "datasets": {
                "era5_1": {
                    "name": "era5-weather",
                    "group_as": "era5",
                    "input": {"join": "not-a-list"},
                },
            },
        },
    }
    result = extract_variables_by_source(config)
    assert "era5" not in result


def test_extract_variables_by_source_weather_skips_non_dict_join_items() -> None:
    """Test that non-dict entries in 'join' are skipped, valid ones still processed."""
    config: dict[str, Any] = {
        "data": {
            "datasets": {
                "era5_1": {
                    "name": "era5-weather",
                    "group_as": "era5",
                    "input": {"join": ["not-a-dict", {"mars": {"param": ["2t"]}}]},
                },
            },
        },
    }
    result = extract_variables_by_source(config)
    assert result["era5"] == ["2t"]


def test_extract_variables_by_source_swallows_attribute_error() -> None:
    """Test that a malformed 'data' section is swallowed rather than raising."""
    config: dict[str, Any] = {"data": "not-a-dict"}
    assert extract_variables_by_source(config) == {}


class TestExtractCadenceFromConfig:
    def test_returns_frequency_for_predict_group_dataset(self) -> None:
        """Test cadence extraction finds the frequency for the predict group's dataset."""
        config: dict[str, Any] = {
            "predict": {"dataset_group": "osisaf-south"},
            "data": {
                "datasets": {
                    "sic1": {
                        "group_as": "osisaf-south",
                        "dates": {"frequency": "1d"},
                    },
                },
            },
        }
        assert extract_cadence_from_config(config) == "1d"

    def test_skips_non_dict_dataset_entries(self) -> None:
        """Test that a non-dict dataset entry is skipped rather than raising."""
        config: dict[str, Any] = {
            "predict": {"dataset_group": "osisaf-south"},
            "data": {
                "datasets": {
                    "bad": "not-a-dict",
                    "sic1": {
                        "group_as": "osisaf-south",
                        "dates": {"frequency": "1d"},
                    },
                },
            },
        }
        assert extract_cadence_from_config(config) == "1d"

    def test_returns_none_when_no_match(self) -> None:
        """Test that no matching dataset returns None."""
        config: dict[str, Any] = {
            "predict": {"dataset_group": "osisaf-south"},
            "data": {"datasets": {}},
        }
        assert extract_cadence_from_config(config) is None

    def test_swallows_attribute_error(self) -> None:
        """Test that a malformed 'predict' section is swallowed rather than raising."""
        config: dict[str, Any] = {"predict": "not-a-dict"}
        assert extract_cadence_from_config(config) is None


class TestExtractTrainingDateRange:
    def test_returns_min_start_and_max_end(self) -> None:
        """Test the training range spans the min start and max end across ranges."""
        config: dict[str, Any] = {
            "data": {
                "split": {
                    "train": [
                        {"start": "2005-01-01", "end": "2010-12-31"},
                        {"start": "2000-01-01", "end": "2003-12-31"},
                    ]
                }
            }
        }
        assert extract_training_date_range(config) == ("2000-01-01", "2010-12-31")

    def test_returns_none_none_when_missing(self) -> None:
        """Test a config without a train split returns (None, None)."""
        assert extract_training_date_range({}) == (None, None)

    def test_swallows_attribute_error(self) -> None:
        """Test that a malformed 'data' section is swallowed rather than raising."""
        config: dict[str, Any] = {"data": "not-a-dict"}
        assert extract_training_date_range(config) == (None, None)


def test_build_metadata_returns_dataclass() -> None:
    """Test that build_metadata returns a Metadata dataclass with extracted fields."""
    config = DictConfig(
        {
            "train": {"trainer": {"max_epochs": 10}},
            "data": {
                "split": {
                    "train": [
                        {"start": "2000-01-01", "end": "2010-12-31"},
                    ]
                },
                "datasets": {
                    "sic1": {
                        "name": "osisaf-sicsouth",
                        "group_as": "osisaf-south",
                    },
                },
            },
            "predict": {"dataset_group": "osisaf-south"},
        }
    )

    metadata = build_metadata(config, model_name="test_model")

    assert isinstance(metadata, Metadata)
    assert metadata.model == "test_model"
    assert metadata.max_epochs == 10
    assert metadata.current_epoch is None
    assert metadata.start == "2000-01-01"
    assert metadata.end == "2010-12-31"
    assert metadata.vars_by_source == {"osisaf-south": ["sea ice"]}


def test_build_metadata_empty_config() -> None:
    """Test build_metadata with empty config returns Metadata with None fields."""
    metadata = build_metadata(DictConfig({}))

    assert isinstance(metadata, Metadata)
    assert metadata.model is None
    assert metadata.max_epochs is None
    assert metadata.current_epoch is None
    assert metadata.start is None
    assert metadata.end is None
    assert metadata.cadence is None
    assert metadata.n_points is None
    assert metadata.vars_by_source is None


def test_build_metadata_extracts_n_history_steps() -> None:
    """Test that build_metadata reads n_history_steps from the predict section."""
    config = DictConfig({"predict": {"n_history_steps": 3}})

    metadata = build_metadata(config)

    assert metadata.n_history_steps == 3


def test_format_metadata_subtitle() -> None:
    """Test format_metadata_subtitle formats Metadata dataclass correctly."""
    metadata = Metadata(
        model="test_model",
        current_epoch=5,
        start="2020-01-01",
        end="2020-01-10",
        cadence="1d",
        n_points=10,
        vars_by_source={"era5": ["2t", "sp"]},
    )

    subtitle = format_metadata_subtitle(metadata)

    assert subtitle is not None
    assert "Model: test_model" in subtitle
    assert "Epoch: 5" in subtitle
    assert "Training Data:" in subtitle
    assert "2020-01-01" in subtitle
    assert "2020-01-10" in subtitle
    assert "10 pts" in subtitle


def test_format_metadata_subtitle_includes_history_window() -> None:
    """Test the subtitle mentions the history window when n_history_steps is set."""
    metadata = Metadata(
        start="2020-01-01",
        end="2020-01-10",
        cadence="1d",
        n_history_steps=3,
    )

    subtitle = format_metadata_subtitle(metadata)

    assert subtitle is not None
    assert "3 day history" in subtitle


def test_format_metadata_subtitle_lists_source_with_no_variables() -> None:
    """Test a source with an empty variable list is listed without parentheses."""
    metadata = Metadata(vars_by_source={"era5": []})

    subtitle = format_metadata_subtitle(metadata)

    assert subtitle is not None
    assert "Training Data: era5" in subtitle
    assert "era5 (" not in subtitle


def test_format_metadata_subtitle_minimal() -> None:
    """Test format_metadata_subtitle with minimal metadata."""
    metadata = Metadata()  # All None

    subtitle = format_metadata_subtitle(metadata)

    assert subtitle is None


def test_plotting_callback_metadata_subtitle_from_config() -> None:
    """Test that PlottingCallback sets metadata_subtitle when config is provided."""
    config = DictConfig(
        {
            "train": {"trainer": {"max_epochs": 5}},
            "data": {
                "split": {
                    "train": [
                        {"start": "2020-01-01", "end": "2020-01-10"},
                    ]
                },
                "datasets": {
                    "sic1": {
                        "name": "osisaf-sicsouth",
                        "group_as": "osisaf-south",
                    },
                },
            },
            "predict": {"dataset_group": "osisaf-south"},
        }
    )

    # Should start with no metadata subtitle
    callback = PlottingCallback()
    assert (
        callback.plotter.plot_spec.metadata_subtitle is None
        or callback.plotter.plot_spec.metadata_subtitle == ""
    )

    # Check that metadata is stored on the callback (subtitle is applied later in make_plots)
    callback.set_metadata(config, model_name="test_model")
    assert callback.plotter_metadata is not None
    assert callback.plotter_metadata.model == "test_model"
    assert callback.plotter_metadata.max_epochs == 5
    assert callback.plotter_metadata.start == "2020-01-01"
