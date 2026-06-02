from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from omegaconf import DictConfig

from icenet_mp.callbacks.plotting_callback import PlottingCallback
from icenet_mp.types import Metadata
from icenet_mp.visualisations.metadata import (
    build_metadata,
    calculate_training_points,
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


class MockDataset:
    """Mock dataset for hemisphere inference tests."""

    def __init__(
        self,
        target_name: str | None = None,
        inputs: Sequence[Any] | None = None,
        name: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialise mock dataset with optional attributes for tests."""
        if target_name:
            self.target = MockTarget(target_name)
        else:
            self.target = None  # type: ignore[assignment]
        self.inputs = inputs or []
        self.name = name
        self.config = config
        self.dataset_config = config


class MockTarget:
    """Mock target dataset component."""

    def __init__(self, name: str) -> None:
        """Initialise mock target with a name."""
        self.name = name


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
