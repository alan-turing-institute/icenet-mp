from collections.abc import Sequence
from functools import cached_property

import numpy as np
from torch.utils.data import Dataset

from icenet_mp.types import ArrayTCHW

from .single_dataset import SingleDataset


class CombinedDataset(Dataset):
    def __init__(
        self,
        datasets: Sequence[SingleDataset],
        target_group_name: str,
        target_variables: Sequence[str],
        *,
        missing_input_value: float | None = None,
        n_forecast_steps: int = 1,
        n_history_steps: int = 1,
    ) -> None:
        """Initialise a combined dataset from a sequence of SingleDatasets.

        One of the datasets must be the target and all must have the same frequency. The
        number of forecast and history steps can be set, which will determine the shape
        of the NTCHW tensors returned by __getitem__.

        When ``missing_input_value`` is set, missing dates inside an input dataset's
        available time range are represented by a full tensor containing that value.
        Forecast target dates are never filled and must always exist. ``None`` preserves
        the existing strict intersection behaviour.
        """
        super().__init__()

        self.missing_input_value = missing_input_value
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps

        self.target = next(
            ds for ds in datasets if ds.name == target_group_name
        ).subset(variables=target_variables)
        self.inputs = list(datasets)

        frequencies = sorted({ds.frequency for ds in datasets})  # type: ignore[type-var]
        if len(frequencies) != 1:
            msg = f"Cannot combine datasets with different frequencies: {frequencies}."
            raise ValueError(msg)
        self.frequency = frequencies[0]

    @cached_property
    def dates(self) -> list[np.datetime64]:
        """Get valid forecast-window start dates."""
        target_date_set = set(self.target.dates)

        if self.missing_input_value is None:
            input_date_set = set.intersection(*(set(ds.dates) for ds in self.inputs))
            candidate_dates = input_date_set
            valid_history = lambda start: all(  # noqa: E731
                date in input_date_set for date in self.get_history_steps(start)
            )
        else:
            # Use the target timeline as the reference timeline. Missing history dates
            # may be filled, but we do not fabricate dates outside any input's observed
            # range or fabricate future target values.
            candidate_dates = target_date_set

            def valid_history(start: np.datetime64) -> bool:
                history_dates = self.get_history_steps(start)
                return all(
                    all(ds.start_date <= date <= ds.end_date for date in history_dates)
                    for ds in self.inputs
                )

        available_dates = sorted(
            available_date
            for available_date in candidate_dates
            if valid_history(available_date)
            and all(
                date in target_date_set
                for date in self.get_forecast_steps(available_date)
            )
        )
        if len(available_dates) == 0:
            msg = (
                "CombinedDataset has no valid dates. This can happen when there "
                "are no valid windows given the configured history/forecast steps or "
                "when the input datasets do not have overlapping time ranges."
            )
            raise ValueError(msg)
        return available_dates

    @property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        return self.dates[-1]

    @property
    def start_date(self) -> np.datetime64:
        """Return the start date of the dataset."""
        return self.dates[0]

    def __len__(self) -> int:
        """Return the total length of the dataset."""
        return len(self.dates)

    def __getitem__(self, idx: int) -> dict[str, ArrayTCHW]:
        """Return one history/forecast window as a dictionary of TCHW arrays."""
        start_date = self.dates[idx]
        history_dates = self.get_history_steps(start_date)

        if self.missing_input_value is None:
            inputs = {
                ds.name: ds.get_tchw_slice(
                    start_date, self.n_history_steps, check=False
                )
                for ds in self.inputs
            }
        else:
            inputs = {
                ds.name: self._input_window_with_missing(ds, history_dates)
                for ds in self.inputs
            }

        return inputs | {
            "target": self.target.get_tchw_slice(
                start_date + self.n_history_steps * self.frequency,
                self.n_forecast_steps,
                check=False,
            )
        }

    def _input_window_with_missing(
        self,
        dataset: SingleDataset,
        dates: Sequence[np.datetime64],
    ) -> ArrayTCHW:
        """Load input dates, replacing missing dates with the configured sentinel."""
        if self.missing_input_value is None:
            msg = "Missing-input filling requested without a configured sentinel value."
            raise RuntimeError(msg)

        available_dates = set(dataset.dates)
        frames = [
            dataset.get_tchw([date])[0]
            if date in available_dates
            else np.full(
                dataset.space.chw,
                self.missing_input_value,
                dtype=np.float32,
            )
            for date in dates
        ]
        return np.stack(frames, axis=0)

    def get_forecast_steps(self, start_date: np.datetime64) -> list[np.datetime64]:
        """Return list of consecutive forecast dates for a given start date."""
        return [
            start_date + (idx + self.n_history_steps) * self.frequency
            for idx in range(self.n_forecast_steps)
        ]

    def get_history_steps(self, start_date: np.datetime64) -> list[np.datetime64]:
        """Return list of consecutive history dates for a given start date."""
        return [
            start_date + idx * self.frequency for idx in range(self.n_history_steps)
        ]
