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
        n_forecast_steps: int = 1,
        n_history_steps: int = 1,
        step_stride: int = 1,
    ) -> None:
        """Initialise a combined dataset from a sequence of SingleDatasets.

        One of the datasets must be the target and all must have the same frequency. The
        number of forecast and history steps can be set, which will determine the shape
        of the NTCHW tensors returned by __getitem__. ``step_stride`` controls the number
        of native dataset timesteps between consecutive history/forecast states. For a
        daily dataset, ``step_stride=7`` therefore produces weekly-spaced model steps.
        """
        super().__init__()

        if step_stride < 1:
            msg = f"step_stride must be at least 1, got {step_stride}."
            raise ValueError(msg)

        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps
        self.step_stride = step_stride

        self.target = next(
            ds for ds in datasets if ds.name == target_group_name
        ).subset(variables=target_variables)
        self.inputs = list(datasets)

        frequencies = sorted({ds.frequency for ds in datasets})  # type: ignore[type-var]
        if len(frequencies) != 1:
            msg = f"Cannot combine datasets with different frequencies: {frequencies}."
            raise ValueError(msg)
        self.frequency = frequencies[0]
        self.step_frequency = self.frequency * self.step_stride

    @cached_property
    def dates(self) -> list[np.datetime64]:
        """Get list of dates that form complete strided history/forecast windows."""
        input_date_set = set.intersection(*(set(ds.dates) for ds in self.inputs))
        target_date_set = set(self.target.dates)
        available_dates = sorted(
            available_date
            for available_date in input_date_set
            if all(
                date in input_date_set
                for date in self.get_history_steps(available_date)
            )
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
        """Return the strided history and forecast tensors for one start date."""
        start_date = self.dates[idx]

        if self.step_stride == 1:
            inputs = {
                ds.name: ds.get_tchw_slice(
                    start_date, self.n_history_steps, check=False
                )
                for ds in self.inputs
            }
            target = self.target.get_tchw_slice(
                start_date + self.n_history_steps * self.frequency,
                self.n_forecast_steps,
                check=False,
            )
        else:
            history_dates = self.get_history_steps(start_date)
            forecast_dates = self.get_forecast_steps(start_date)
            inputs = {ds.name: ds.get_tchw(history_dates) for ds in self.inputs}
            target = self.target.get_tchw(forecast_dates)

        return inputs | {"target": target}

    def get_forecast_steps(self, start_date: np.datetime64) -> list[np.datetime64]:
        """Return forecast dates at the configured temporal stride."""
        return [
            start_date + (idx + self.n_history_steps) * self.step_frequency
            for idx in range(self.n_forecast_steps)
        ]

    def get_history_steps(self, start_date: np.datetime64) -> list[np.datetime64]:
        """Return history dates at the configured temporal stride."""
        return [
            start_date + idx * self.step_frequency
            for idx in range(self.n_history_steps)
        ]
