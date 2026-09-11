from typing import cast

import numpy as np
import pytest

from icenet_mp.data import CombinedDataset, SingleDataset
from icenet_mp.types import Metadata
from icenet_mp.visualisations.metadata_builder import MetadataBuilder

builder = MetadataBuilder()
format_metadata_subtitle = builder.format_subtitle


def fake_single_dataset(name: str, variable_names: list[str]) -> SingleDataset:
    """Return a duck-typed SingleDataset stand-in exposing name/variable_names."""

    class FakeSingleDataset:
        def __init__(self, name: str, variable_names: list[str]) -> None:
            self.name = name
            self.variable_names = variable_names

    return cast("SingleDataset", FakeSingleDataset(name, variable_names))


def fake_combined_dataset(
    *,
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-10",
    frequency: np.timedelta64 | None = None,
    length: int = 10,
    n_history_steps: int = 0,
    inputs: list[SingleDataset] | None = None,
) -> CombinedDataset:
    """Return a duck-typed CombinedDataset stand-in exposing what from_dataset reads."""

    class FakeCombinedDataset:
        def __init__(self) -> None:
            self.start_date = np.datetime64(start_date)
            self.end_date = np.datetime64(end_date)
            self.frequency = (
                frequency if frequency is not None else np.timedelta64(1, "D")
            )
            self.n_history_steps = n_history_steps
            self.inputs = inputs if inputs is not None else []

        def __len__(self) -> int:
            return length

    return cast("CombinedDataset", FakeCombinedDataset())


class TestFormatCadence:
    @pytest.mark.parametrize(
        ("hours", "expected"),
        [
            (24, "daily"),
            (48, "2d"),
            (72, "3d"),
            (1, "hourly"),
            (6, "6h"),
            (0.5, "0.5h"),
        ],
    )
    def test_formats_hours_as_cadence_label(self, hours: float, expected: str) -> None:
        """Format a dataset's frequency into a short, human-readable cadence label."""
        frequency = np.timedelta64(int(hours * 60), "m")
        assert MetadataBuilder._format_cadence(frequency) == expected


class TestBuildFromDataset:
    def test_derives_dates_cadence_and_length_from_dataset(self) -> None:
        """Metadata fields come from the dataset's realised state, not from config."""
        dataset = fake_combined_dataset(
            start_date="2020-01-01T12:30:00",
            end_date="2020-01-10T00:00:00",
            frequency=np.timedelta64(1, "D"),
            length=10,
            n_history_steps=3,
        )

        metadata = builder.from_dataset(dataset, current_epoch=5, model_name="unet")

        assert metadata.model == "unet"
        assert metadata.current_epoch == 5
        assert metadata.start == "2020-01-01"
        assert metadata.end == "2020-01-10"
        assert metadata.cadence == "daily"
        assert metadata.n_points == 10
        assert metadata.n_history_steps == 3

    def test_defaults_model_and_epoch_to_none(self) -> None:
        """Omitted model_name/current_epoch fall back to None."""
        dataset = fake_combined_dataset()

        metadata = builder.from_dataset(dataset)

        assert metadata.model is None
        assert metadata.current_epoch is None

    def test_collects_sorted_variable_names_by_source(self) -> None:
        """vars_by_source maps each input dataset's name to its sorted variable names."""
        dataset = fake_combined_dataset(
            inputs=[
                fake_single_dataset("era5", ["sp", "2t"]),
                fake_single_dataset("osisaf-south", ["sic"]),
            ]
        )

        metadata = builder.from_dataset(dataset)

        assert metadata.vars_by_source == {
            "era5": ["2t", "sp"],
            "osisaf-south": ["sic"],
        }

    def test_empty_inputs_yields_none_vars_by_source(self) -> None:
        """No input datasets means no variable-by-source mapping."""
        dataset = fake_combined_dataset(inputs=[])

        metadata = builder.from_dataset(dataset)

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
    assert "3 step history" in subtitle


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
