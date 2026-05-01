from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import numpy as np
from anemoi.datasets.data import open_dataset
from anemoi.datasets.data.dataset import Dataset as AnemoiDataset
from torch.utils.data import Dataset

from icenet_mp.types import ArrayCHW, ArrayTCHW, DataSpace, Hemisphere
from icenet_mp.utils import normalise_date


class SingleDataset(Dataset):
    def __init__(
        self,
        name: str,
        input_files: list[Path],
        *,
        date_ranges: Sequence[dict[str, str | None]] = [{"start": None, "end": None}],
        variables: Sequence[str] = (),
    ) -> None:
        """A dataset for use by IceNet-MP.

        The underlying Anemoi dataset has shape [T; C; ensembles; position].
        We reshape this to CHW before returning.
        """
        super().__init__()
        self._datasets: list[AnemoiDataset] = []
        self._date_ranges = sorted(
            date_ranges, key=lambda dr: "" if dr["start"] is None else dr["start"]
        )
        self.hemisphere: Hemisphere = (
            "north"
            if any("north" in str(input_file).lower() for input_file in input_files)
            else "south"
        )
        self._input_files = input_files
        self._name = name
        self._variables = set(variables)

    @cached_property
    def _date2idx(self) -> dict[np.datetime64, int]:
        """Map date to global index in the dataset."""
        return {date: idx for idx, date in enumerate(self.dates)}

    @cached_property
    def _idx2anemoi(self) -> dict[int, tuple[int, int]]:
        """Map global index to a location in an Anemoi dataset."""
        idx2anemoi = {}
        for idx_ds, dataset in enumerate(self.datasets):
            for idx_date, date in enumerate(dataset.dates):
                idx_global = self._date2idx.get(normalise_date(date), None)
                if idx_global is not None:
                    idx2anemoi[idx_global] = (idx_ds, idx_date)
        return idx2anemoi

    @property
    def datasets(self) -> list[AnemoiDataset]:
        """Load one or more underlying Anemoi datasets.

        Each date range results in a separate dataset.
        """
        if not self._datasets:
            for date_range in self._date_ranges:
                # Set the time range for this dataset
                ds_kwargs: dict[str, str | set[str] | None] = {
                    "start": date_range["start"],
                    "end": date_range["end"],
                }
                # Select a subset of variables if specified
                if self._variables:
                    ds_kwargs["select"] = self._variables
                _dataset = open_dataset(self._input_files, **ds_kwargs)
                _dataset._name = self._name
                self._datasets.append(_dataset)
        return self._datasets

    @cached_property
    def dates(self) -> list[np.datetime64]:
        """Return all available dates in the dataset, removing any that are missing."""
        return sorted(
            {
                normalise_date(date)
                for ds in self.datasets
                for date in np.delete(ds.dates, list(ds.missing))
            }
        )

    @cached_property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        return self.dates[-1]

    @property
    def frequency(self) -> np.timedelta64:
        """Return the frequency of the dataset."""
        return np.timedelta64(self.datasets[0].frequency)

    @cached_property
    def latitudes(self) -> list[float]:
        """Return the latitudes of the dataset."""
        reference_latitudes = self.datasets[0].latitudes
        n_different = sum(
            not np.array_equal(ds.latitudes, reference_latitudes)
            for ds in self.datasets
        )
        if n_different != 0:
            msg = f"All date ranges must have the same latitudes, found {n_different + 1} different values"
            raise ValueError(msg)
        return reference_latitudes.tolist()

    @cached_property
    def longitudes(self) -> list[float]:
        """Return the longitudes of the dataset."""
        reference_longitudes = self.datasets[0].longitudes
        n_different = sum(
            not np.array_equal(ds.longitudes, reference_longitudes)
            for ds in self.datasets
        )
        if n_different != 0:
            msg = f"All date ranges must have the same longitudes, found {n_different + 1} different values"
            raise ValueError(msg)
        return reference_longitudes.tolist()

    @property
    def name(self) -> str:
        """Return the name of the dataset."""
        return self._name

    @cached_property
    def space(self) -> DataSpace:
        """Return the data space for this dataset."""
        # Check all datasets have the same number of channels
        per_ds_channels = sorted({ds.shape[1] for ds in self.datasets})
        if len(per_ds_channels) != 1:
            msg = f"All date ranges must have the same number of channels, found {len(per_ds_channels)} different values"
            raise ValueError(msg)
        # Check all datasets have the same shape
        per_ds_shape = sorted({ds.field_shape for ds in self.datasets})
        if len(per_ds_shape) != 1:
            msg = f"All date ranges must have the same shape, found {len(per_ds_shape)} different values"
            raise ValueError(msg)
        # Return the data space
        return DataSpace(
            channels=per_ds_channels[0],
            name=self.name,
            shape=per_ds_shape[0],
        )

    @property
    def start_date(self) -> np.datetime64:
        """Return the start date of the dataset."""
        return self.dates[0]

    @cached_property
    def variable_names(self) -> list[str]:
        """Return the variable names for this dataset.

        The variable names are extracted from the underlying Anemoi dataset.
        All datasets must have the same variables.
        """
        variable_names = {tuple(sorted(ds.variables)) for ds in self.datasets}
        if len(variable_names) != 1:
            msg = f"All date ranges must have the same variables, found {len(variable_names)} different values."
            raise ValueError(msg)
        return self.datasets[0].variables

    def __len__(self) -> int:
        """Return the total length of the dataset."""
        return len(self.dates)

    def __getitem__(self, idx: int) -> ArrayCHW:
        """Return the data for a single timestep in [C, H, W] format."""
        try:
            idx_ds, idx_date = self._idx2anemoi[idx]
            return self.datasets[idx_ds][idx_date].reshape(self.space.chw)
        except KeyError as exc:
            msg = f"Index {idx} out of range for dataset of length {len(self)}."
            raise IndexError(msg) from exc

    def get_tchw(self, dates: Sequence[np.datetime64]) -> ArrayTCHW:
        """Return the data for a series of timesteps in [T, C, H, W] format."""
        return np.stack(
            [self[self.to_index(target_date)] for target_date in dates], axis=0
        )

    def subset(self, variables: Sequence[str]) -> "SingleDataset":
        return SingleDataset(
            name=self.name,
            input_files=self._input_files,
            date_ranges=self._date_ranges,
            variables=variables,
        )

    def to_index(self, date: np.datetime64) -> int:
        """Return the index of a given date in the dataset."""
        try:
            return self._date2idx[date]
        except KeyError as exc:
            msg = f"Date {date} not found in the dataset {self.start_date} to {self.end_date} every {self.frequency}"
            raise IndexError(msg) from exc
