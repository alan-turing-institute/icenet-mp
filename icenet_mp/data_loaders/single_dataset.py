from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import numpy as np
from anemoi.datasets.data import open_dataset
from anemoi.datasets.data.dataset import Dataset as AnemoiDataset
from torch.utils.data import Dataset

from icenet_mp.types import ArrayCHW, ArrayTCHW, DataSpace, Hemisphere
from icenet_mp.utils import normalise_date


class SingleDataset(Dataset):
    """A dataset containing one or more timeslices of data from a single source."""

    # Cache anemoi reads at class level to reduce disk I/O
    anemoi_cache: ClassVar[dict[tuple[Path, ...], AnemoiDataset]] = {}

    def __init__(
        self,
        name: str,
        input_files: Sequence[Path],
        *,
        date_ranges: Sequence[dict[str, str | None]] = [{"start": None, "end": None}],
        variables: Sequence[str] = (),
    ) -> None:
        """A dataset for use by IceNet-MP.

        The underlying Anemoi dataset has shape [T; C; ensembles; position].
        We reshape this to CHW before returning.
        """
        super().__init__()
        self._date_ranges = sorted(
            date_ranges, key=lambda dr: "" if dr["start"] is None else dr["start"]
        )
        self.hemisphere: Hemisphere = (
            "north"
            if any("north" in str(input_file).lower() for input_file in input_files)
            else "south"
        )
        self._input_files = tuple(sorted(input_files))
        self._name = name
        self._variables = set(variables)

    @classmethod
    def load_dataset(cls, input_files: tuple[Path, ...]) -> AnemoiDataset:
        if input_files not in cls.anemoi_cache:
            cls.anemoi_cache[input_files] = open_dataset(input_files)
        return cls.anemoi_cache[input_files]

    @cached_property
    def _date2idx(self) -> dict[np.datetime64, int]:
        """Map date to global index in the dataset."""
        return {date: idx for idx, date in enumerate(self.dates)}

    @cached_property
    def _idx2anemoi(self) -> dict[int, tuple[int, int]]:
        """Map global index to a location in an Anemoi dataset."""
        idx2anemoi = {}
        for idx_ds, dataset in enumerate(self.dataslices):
            for idx_date, date in enumerate(dataset.dates):
                idx_global = self._date2idx.get(normalise_date(date), None)
                if idx_global is not None:
                    idx2anemoi[idx_global] = (idx_ds, idx_date)
        return idx2anemoi

    @cached_property
    def dataslices(self) -> list[AnemoiDataset]:
        """Get all slices of contiguous dates from the underlying Anemoi dataset."""
        return [
            self.load_dataset(self._input_files)._subset(
                name=self._name,
                start=date_range["start"],
                end=date_range["end"],
                **({"select": self._variables} if self._variables else {}),
            )
            for date_range in self._date_ranges
        ]

    @cached_property
    def dates(self) -> list[np.datetime64]:
        """Return all available dates in the dataset, removing any that are missing."""
        return sorted(
            {
                normalise_date(date)
                for ds in self.dataslices
                for date in np.delete(ds.dates, list(ds.missing))
            }
        )

    @cached_property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        return self.dates[-1]

    @cached_property
    def frequency(self) -> np.timedelta64:
        """Return the frequency of the dataset."""
        return np.timedelta64(self.dataslices[0].frequency)

    @cached_property
    def latitudes(self) -> list[float]:
        """Return the latitudes of the dataset."""
        return self.dataslices[0].latitudes.tolist()

    @cached_property
    def longitudes(self) -> list[float]:
        """Return the longitudes of the dataset."""
        return self.dataslices[0].longitudes.tolist()

    @cached_property
    def name(self) -> str:
        """Return the name of the dataset."""
        return self._name

    @cached_property
    def space(self) -> DataSpace:
        """Return the data space for this dataset."""
        return DataSpace(
            channels=self.dataslices[0].shape[1],
            name=self.name,
            shape=self.dataslices[0].field_shape,
        )

    @cached_property
    def start_date(self) -> np.datetime64:
        """Return the start date of the dataset."""
        return self.dates[0]

    @cached_property
    def variable_names(self) -> list[str]:
        """Return the variable names for this dataset."""
        return self.dataslices[0].variables

    def __len__(self) -> int:
        """Return the total length of the dataset."""
        return len(self.dates)

    def __getitem__(self, idx: int) -> ArrayCHW:
        """Return the data for a single timestep in [C, H, W] format."""
        try:
            idx_ds, idx_date = self._idx2anemoi[idx]
            return self.dataslices[idx_ds][idx_date].reshape(self.space.chw)
        except KeyError as exc:
            msg = f"Index {idx} out of range for dataset of length {len(self)}."
            raise IndexError(msg) from exc

    def get_tchw(self, dates: Sequence[np.datetime64]) -> ArrayTCHW:
        """Return the data for an arbitrary sequence of timesteps in [T, C, H, W] format."""
        return np.stack(
            [self[self.to_index(target_date)] for target_date in dates], axis=0
        )

    def get_tchw_slice(
        self, start_date: np.datetime64, n_steps: int, *, check: bool = True
    ) -> ArrayTCHW:
        """Return the data for consecutive timesteps in [T, C, H, W] format.

        Since contiguous dates must be in a single dataslice, we simply identify which
        one this is and read from it.

        If `check` is True then we check that we're not crossing the boundary between
        dataslices, adding a small amount of overhead.

        Args:
            start_date: The date of the first timestep to return.
            n_steps: The number of consecutive timesteps to return.
            check: Whether to check that the requested slice is valid. If False, this
                   method may return meaningless or incorrect data if the requested
                   slice is invalid

        """
        try:
            idx_global_start = self._date2idx[start_date]
            idx_ds_start, idx_date_start = self._idx2anemoi[idx_global_start]
            if check:
                idx_global_end = idx_global_start + n_steps - 1
                idx_ds_end, _ = self._idx2anemoi[idx_global_end]
                if idx_ds_start != idx_ds_end:
                    msg = (
                        f"Requested slice of {n_steps} steps following {start_date} "
                        f"crosses the boundary between dataslices {idx_ds_start} and "
                        f"{idx_ds_end}."
                    )
                    raise ValueError(msg)
            dataslice = self.dataslices[idx_ds_start][
                idx_date_start : idx_date_start + n_steps
            ]
            return dataslice.reshape(n_steps, *self.space.chw)
        except KeyError as exc:
            msg = (
                f"Requested slice of {n_steps} steps following {start_date} "
                f"is out of range for dataset with dates from {self.start_date} to "
                f"{self.end_date}"
            )
            raise ValueError(msg) from exc

    def subset(
        self,
        *,
        date_ranges: Sequence[dict[str, str | None]] | None = None,
        variables: Sequence[str] | None = None,
    ) -> "SingleDataset":
        return SingleDataset(
            name=self.name,
            input_files=self._input_files,
            date_ranges=date_ranges or self._date_ranges,
            variables=variables or list(self._variables),
        )

    def to_index(self, date: np.datetime64) -> int:
        """Return the index of a given date in the dataset."""
        try:
            return self._date2idx[date]
        except KeyError as exc:
            msg = (
                f"Date {np.datetime_as_string(date, unit='D')} not found in the "
                f"dataset {self.start_date} to {self.end_date} every {self.frequency}"
            )
            raise IndexError(msg) from exc
