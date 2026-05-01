from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from icenet_mp.data_loaders.single_dataset import SingleDataset
from icenet_mp.types import DataSpace


class MockAnemoiDataset:
    def __init__(self, channels: int, height: int, width: int) -> None:
        """A mock Anemoi dataset for testing purposes."""
        self.shape = (1, channels, height * width)
        self.field_shape = (height, width)


class TestSingleDataset:
    dates_str = ("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05")
    dates_np = tuple(np.datetime64(s) for s in dates_str)

    def test_name(self) -> None:
        dataset = SingleDataset(
            name="test_dataset",
            input_files=[],
        )
        assert dataset.name == "test_dataset"

    def test_dates(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        # Test dates
        assert all(date in dataset.dates for date in self.dates_np)

    def test_end_date(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            date_ranges=[{"start": None, "end": self.dates_str[1]}],
        )
        assert dataset.start_date == self.dates_np[0]
        assert dataset.end_date == self.dates_np[1]

    def test_date_ranges(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            date_ranges=[
                {"start": self.dates_str[0], "end": self.dates_str[1]},
                {"start": self.dates_str[-2], "end": self.dates_str[-1]},
            ],
        )
        assert self.dates_np[2] not in dataset.dates
        assert len(dataset.datasets) == 2
        assert len(dataset) == 4

    def test_missing_dates(self, mock_dataset_missing_dates: Path) -> None:
        """Test that missing dates are excluded from SingleDataset.dates."""
        # Create SingleDataset with indices 1 and 3 (2020-01-02 and 2020-01-04) missing
        dataset = SingleDataset(
            name="test_missing", input_files=[mock_dataset_missing_dates]
        )

        # Check that missing dates are excluded
        missing_indices = {1, 3}
        expected_dates = [
            date for idx, date in enumerate(self.dates_np) if idx not in missing_indices
        ]
        assert len(expected_dates) == 3

        assert dataset.dates == expected_dates
        assert self.dates_np[1] not in dataset.dates  # 2020-01-02 should be missing
        assert self.dates_np[3] not in dataset.dates  # 2020-01-04 should be missing

    def test_start_date(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            date_ranges=[{"start": self.dates_str[1], "end": None}],
        )
        assert dataset.start_date == self.dates_np[1]
        assert dataset.end_date == self.dates_np[-1]

    def test_datetime_normalization(
        self, mock_dataset_non_normalized_times: Path
    ) -> None:
        """Test that datetime normalization is applied to all dates."""
        dataset = SingleDataset(
            name="test_normalized",
            input_files=[mock_dataset_non_normalized_times],
        )

        # All dates should be normalized to 00:00:00
        for date in dataset.dates:
            dt: datetime = date.astype("datetime64[us]").astype(datetime)
            assert dt.hour == 0
            assert dt.minute == 0
            assert dt.second == 0
            assert dt.microsecond == 0

        # Check specific normalized dates
        expected_dates = [
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-02"),
            np.datetime64("2020-01-03"),
            np.datetime64("2020-01-04"),
            np.datetime64("2020-01-05"),
        ]
        assert dataset.dates == expected_dates

    def test_getitem(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        # Check return type and shape
        data_array = dataset[0]
        assert isinstance(data_array, np.ndarray)
        assert data_array.shape == (3, 2, 2)
        # Check exception for out of range
        with pytest.raises(
            IndexError, match="Index 10 out of range for dataset of length 5"
        ):
            dataset[10]

    def test_get_tchw(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        # Check return type and shape
        data_array = dataset.get_tchw(self.dates_np)
        assert isinstance(data_array, np.ndarray)
        assert data_array.shape == (5, 3, 2, 2)
        # Check exception for out of range
        with pytest.raises(
            IndexError, match="Date 1970-01-01 not found in the dataset"
        ):
            dataset.get_tchw([np.datetime64("1970-01-01"), np.datetime64("1970-01-02")])

    def test_get_tchw_with_missing_dates(
        self, mock_dataset_missing_dates: Path
    ) -> None:
        """Test that get_tchw works correctly when dates are missing."""
        dataset = SingleDataset(
            name="test_missing", input_files=[mock_dataset_missing_dates]
        )

        # Get TCHW for available dates
        missing_indices = {1, 3}
        expected_dates = [
            date for idx, date in enumerate(self.dates_np) if idx not in missing_indices
        ]
        assert len(expected_dates) == 3

        # Result should have shape (3, C, H, W)
        result = dataset.get_tchw(expected_dates)
        assert result.shape == (3, 1, 2, 2)

        # Attempting to get TCHW for missing dates should raise IndexError
        with pytest.raises(
            IndexError, match="Date 2020-01-02 not found in the dataset"
        ):
            dataset.get_tchw([self.dates_np[1]])

    def test_len(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        assert len(dataset) == 5

    def test_len_with_missing_dates(self, mock_dataset_missing_dates: Path) -> None:
        """Test that dataset length reflects missing dates."""
        dataset = SingleDataset(
            name="test_missing", input_files=[mock_dataset_missing_dates]
        )
        # There should be 5 dates with 2 missing
        assert len(dataset) == 3

    def test_space(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        # Test data space
        assert isinstance(dataset.space, DataSpace)
        assert dataset.space.channels == 3
        assert dataset.space.shape == (2, 2)

    def test_space_error_shape(self) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[],
        )
        dataset._datasets = [  # type: ignore[reportAttributeAccessIssue]
            MockAnemoiDataset(1, 32, 32),
            MockAnemoiDataset(1, 32, 64),
        ]
        # Test data space shapes
        with pytest.raises(
            ValueError,
            match="All date ranges must have the same shape, found 2 different values",
        ):
            _ = dataset.space

    def test_space_error_channels(self) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[],
        )
        dataset._datasets = [  # type: ignore[reportAttributeAccessIssue]
            MockAnemoiDataset(10, 32, 32),
            MockAnemoiDataset(11, 32, 32),
        ]
        # Test data space channels
        with pytest.raises(
            ValueError,
            match="All date ranges must have the same number of channels, found 2 different values",
        ):
            _ = dataset.space

    def test_subset(self, mock_dataset: Path) -> None:
        """Test the select_variables classmethod."""
        # Create a dataset with all variables
        original_dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        assert original_dataset.space.channels == 3

        # Use select_variables to create a subset
        subset_dataset = original_dataset.subset(variables=["ice_conc"])
        assert subset_dataset.space.channels == 1
        assert subset_dataset.name == "mock_dataset"

        # Check that the data shape is correct
        data_array = subset_dataset[0]
        assert data_array.shape == (1, 2, 2)

    def test_subset_preserves_date_ranges(self, mock_dataset: Path) -> None:
        """Test that select_variables preserves date ranges."""
        # Create a dataset with date ranges
        original_dataset = SingleDataset(
            name="mock_dataset_multi",
            input_files=[mock_dataset],
            date_ranges=[{"start": self.dates_str[0], "end": self.dates_str[2]}],
        )

        # Use select_variables to create a subset
        subset_dataset = original_dataset.subset(variables=["ice_thickness"])

        # Check that date ranges are preserved
        assert subset_dataset.start_date == self.dates_np[0]
        assert subset_dataset.end_date == self.dates_np[2]
        assert len(subset_dataset) == 3

    def test_to_index(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        # Check known dates
        assert dataset.to_index(self.dates_np[0]) == 0
        assert dataset.to_index(self.dates_np[3]) == 3
        # Check exception for out of range
        with pytest.raises(
            IndexError, match="Date 1970-01-01 not found in the dataset"
        ):
            dataset.to_index(np.datetime64("1970-01-01"))

    def test_to_index_with_missing_dates(
        self, mock_dataset_missing_dates: Path
    ) -> None:
        """Test that to_index works correctly when dates are missing."""
        dataset = SingleDataset(
            name="test_missing", input_files=[mock_dataset_missing_dates]
        )

        # Indices should be mapped to available dates only
        assert dataset.to_index(self.dates_np[0]) == 0  # 2020-01-01
        assert dataset.to_index(self.dates_np[2]) == 1  # 2020-01-03
        assert dataset.to_index(self.dates_np[4]) == 2  # 2020-01-05

        # Missing dates should raise IndexError
        with pytest.raises(
            IndexError, match="Date 2020-01-02 not found in the dataset"
        ):
            dataset.to_index(self.dates_np[1])
        with pytest.raises(
            IndexError, match="Date 2020-01-04 not found in the dataset"
        ):
            dataset.to_index(self.dates_np[3])

    def test_variable_selection_all(self, mock_dataset: Path) -> None:
        """Test selecting all variables from a multi-variable dataset."""
        dataset = SingleDataset(
            name="mock_dataset_multi",
            input_files=[mock_dataset],
            variables=["ice_conc", "ice_thickness", "temperature"],
        )
        # Should have 3 channels
        assert dataset.space.channels == 3
        # Check data shape
        data_array = dataset[0]
        assert data_array.shape == (3, 2, 2)

    def test_variable_selection_multiple(self, mock_dataset: Path) -> None:
        """Test selecting multiple variables from a multi-variable dataset."""
        dataset = SingleDataset(
            name="mock_dataset_multi",
            input_files=[mock_dataset],
            variables=["ice_conc", "temperature"],
        )
        # Should have 2 channels
        assert dataset.space.channels == 2
        # Check data shape
        data_array = dataset[0]
        assert data_array.shape == (2, 2, 2)

    def test_variable_selection_none(self, mock_dataset: Path) -> None:
        """Test that not specifying variables loads all variables."""
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        # Should have all 3 channels (ice_conc, ice_thickness and temperature)
        assert dataset.space.channels == 3

    def test_variable_selection_single(self, mock_dataset: Path) -> None:
        """Test selecting a single variable from a multi-variable dataset."""
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            variables=["ice_conc"],
        )
        # Should have only 1 channel
        assert dataset.space.channels == 1
        # Check data shape
        data_array = dataset[0]
        assert data_array.shape == (1, 2, 2)
        assert dataset.start_date == self.dates_np[0]
        assert dataset.end_date == self.dates_np[-1]
