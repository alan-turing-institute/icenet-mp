from pathlib import Path

import numpy as np
import pytest

from icenet_mp.data_loaders.combined_dataset import CombinedDataset
from icenet_mp.data_loaders.single_dataset import SingleDataset


class TestCombinedDataset:
    dates_str = ("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05")
    dates_np = tuple(np.datetime64(f"{s}T12:00:00") for s in dates_str)

    def test_no_valid_dates_non_overlapping_ranges(self, mock_dataset: Path) -> None:
        """Test that CombinedDataset raises ValueError when datasets have non-overlapping date ranges."""
        # Create and combine two datasets with non-overlapping date ranges
        dataset1 = SingleDataset(
            name="dataset1",
            input_files=[mock_dataset],
            date_ranges=[{"start": self.dates_str[0], "end": self.dates_str[1]}],
        )
        dataset2 = SingleDataset(
            name="dataset2",
            input_files=[mock_dataset],
            date_ranges=[{"start": self.dates_str[3], "end": self.dates_str[4]}],
        )
        combined = CombinedDataset(
            datasets=[dataset1, dataset2],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=1,
        )

        # Confirm that no valid dates are available
        with pytest.raises(ValueError, match="CombinedDataset has no valid dates"):
            _ = combined.dates

    def test_no_valid_dates_insufficient_history_steps(
        self, mock_dataset: Path
    ) -> None:
        """Test that CombinedDataset raises ValueError when history steps exceed available dates."""
        dataset = SingleDataset(
            name="dataset1",
            input_files=[mock_dataset],
            date_ranges=[{"start": self.dates_str[0], "end": self.dates_str[1]}],
        )

        # Create combined dataset with history steps larger than available dates
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=10,  # Only 2 dates available
            n_forecast_steps=1,
        )

        # Confirm that no valid dates are available
        with pytest.raises(ValueError, match="CombinedDataset has no valid dates"):
            _ = combined.dates

    def test_no_valid_dates_insufficient_forecast_steps(
        self, mock_dataset: Path
    ) -> None:
        """Test that CombinedDataset raises ValueError when forecast steps exceed available dates."""
        dataset = SingleDataset(
            name="dataset1",
            input_files=[mock_dataset],
            date_ranges=[{"start": self.dates_str[0], "end": self.dates_str[1]}],
        )

        # Create combined dataset with forecast steps larger than available dates
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=10,  # Only 2 dates available
        )

        # Confirm that no valid dates are available
        with pytest.raises(ValueError, match="CombinedDataset has no valid dates"):
            _ = combined.dates

    def test_no_valid_dates_incompatible_times(self, mock_dataset: Path) -> None:
        """Test that CombinedDataset raises ValueError when datasets have incompatible times."""
        dataset = SingleDataset(
            name="dataset",
            input_files=[mock_dataset],
        )
        # Apply an offset of 12h to the dates in a second dataset
        dataset_offset = SingleDataset(
            name="dataset_offset",
            input_files=[mock_dataset],
        )
        dataset_offset.dates = [
            date + np.timedelta64(720, "m") for date in self.dates_np
        ]

        # Create combined dataset
        combined = CombinedDataset(
            datasets=[dataset, dataset_offset],
            target_group_name="dataset",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=1,
        )

        # Confirm that no valid dates are available
        with pytest.raises(ValueError, match="CombinedDataset has no valid dates"):
            _ = combined.dates

    def test_valid_dates(self, mock_dataset: Path) -> None:
        """Test that CombinedDataset works correctly with valid overlapping dates."""
        # Create and combine two datasets with overlapping date ranges
        dataset1 = SingleDataset(
            name="dataset1",
            input_files=[mock_dataset],
        )
        dataset2 = SingleDataset(
            name="dataset2",
            input_files=[mock_dataset],
        )
        combined = CombinedDataset(
            datasets=[dataset1, dataset2],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=2,
            n_forecast_steps=1,
        )

        # Should not raise an error
        dates = combined.dates
        assert len(dates) > 0
        assert all(isinstance(date, np.datetime64) for date in dates)

    def test_start_and_end_dates(self, mock_dataset: Path) -> None:
        """Test that start_date and end_date are correctly calculated from available dates."""
        dataset1 = SingleDataset(
            name="dataset1",
            input_files=[mock_dataset],
            date_ranges=[{"start": self.dates_str[0], "end": self.dates_str[-2]}],
        )
        dataset2 = SingleDataset(
            name="dataset2",
            input_files=[mock_dataset],
            date_ranges=[{"start": self.dates_str[1], "end": self.dates_str[-1]}],
        )
        combined = CombinedDataset(
            datasets=[dataset1, dataset2],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=1,
        )

        assert combined.start_date == combined.dates[0]
        assert combined.end_date == combined.dates[-1]

    def test_getitem(self, mock_dataset: Path) -> None:
        """__getitem__ returns a dict with the expected keys and array shapes."""
        dataset = SingleDataset(name="dataset1", input_files=[mock_dataset])
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=2,
            n_forecast_steps=1,
        )
        batch = combined[0]
        assert isinstance(batch, dict)
        assert set(batch.keys()) == {"dataset1", "target"}
        # input has shape [n_history_steps, C_input, H, W]
        assert batch["dataset1"].shape == (2, 3, 2, 2)
        # target has shape (n_forecast_steps, C_target, H, W)
        assert batch["target"].shape == (1, 1, 2, 2)

    def test_getitem_time_offset(self, mock_dataset: Path) -> None:
        """__getitem__ places the target window n_history_steps ahead of the input window."""
        dataset = SingleDataset(name="dataset1", input_files=[mock_dataset])
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=2,
            n_forecast_steps=1,
        )
        # With start_date 2020-01-01, n_history_steps=2 and n_forecast_steps=1 we expect
        # history covers 2020-01-01 to 2020-01-02
        # target covers 2020-01-03
        batch = combined[0]
        np.testing.assert_array_equal(
            batch["dataset1"],
            combined.inputs[0].get_tchw([self.dates_np[0], self.dates_np[1]]),
        )
        np.testing.assert_array_equal(
            batch["target"],
            combined.target.get_tchw([self.dates_np[2]]),
        )

    def test_lazy_loading(self, mock_dataset: Path) -> None:
        """Test that dates are lazy-loaded and cached."""
        dataset = SingleDataset(
            name="dataset1",
            input_files=[mock_dataset],
        )

        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=1,
        )

        # Initially dates should be unavailable
        assert "dates" not in combined.__dict__

        # Access dates for the first time
        dates1 = combined.dates
        assert "dates" in combined.__dict__
        assert combined.__dict__["dates"] is dates1

        # Access dates again, should return cached value
        dates2 = combined.dates
        assert dates1 is dates2
