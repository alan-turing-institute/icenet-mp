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
        target_uncertainty_variable: str | None = None,
    ) -> None:
        """Initialise a combined dataset from a sequence of SingleDatasets.

        One of the datasets must be the target and all must have the same frequency. The
        number of forecast and history steps can be set, which will determine the shape
        of the NTCHW tensors returned by __getitem__.

        If ``target_uncertainty_variable`` is set, the corresponding variable is read
        for the same forecast dates as the target and returned under the
        ``target_uncertainty`` key. Uncertainty is kept in its ingested physical scale
        rather than min-max normalised so invalid sentinels and uncertainty thresholds
        retain their meaning.
        """
        super().__init__()

        # Store the number of forecast and history steps
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps

        # Create a new dataset for the target with only the selected variables
        target_source = next(ds for ds in datasets if ds.name == target_group_name)
        self.target = target_source.subset(variables=target_variables)
        self.target_uncertainty: SingleDataset | None = None
        if target_uncertainty_variable is not None:
            if len(target_variables) != 1:
                msg = (
                    "Uncertainty-weighted targets currently require exactly one "
                    f"predicted variable, found {list(target_variables)}."
                )
                raise ValueError(msg)
            if target_uncertainty_variable not in target_source.variable_names:
                msg = (
                    f"Target uncertainty variable {target_uncertainty_variable!r} was "
                    f"not found in dataset {target_group_name!r}. Available variables: "
                    f"{target_source.variable_names}."
                )
                raise ValueError(msg)
            self.target_uncertainty = target_source.subset(
                variables=[target_uncertainty_variable]
            )
            # This auxiliary variable is observational sigma, not a model input. Keep
            # the ingested physical fraction (and sentinel values) intact rather than
            # applying SingleDataset's model-input min-max normalisation.
            self.target_uncertainty._normalise = False

        self.inputs = list(datasets)

        # Require that all datasets have the same frequency
        frequencies = sorted({ds.frequency for ds in datasets})  # type: ignore[type-var]
        if len(frequencies) != 1:
            msg = f"Cannot combine datasets with different frequencies: {frequencies}."
            raise ValueError(msg)
        self.frequency = frequencies[0]

    @cached_property
    def dates(self) -> list[np.datetime64]:
        """Get list of dates that are available in all datasets."""
        # Identify dates that exist in all input datasets
        input_date_set = set.intersection(*(set(ds.dates) for ds in self.inputs))
        target_date_set = set(self.target.dates)
        available_dates = sorted(
            available_date
            for available_date in input_date_set
            # ... if all inputs have n_history_steps starting on start_date
            if all(
                date in input_date_set
                for date in self.get_history_steps(available_date)
            )
            # ... and if the target has n_forecast_steps starting after the history dates
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
        """Return the data for a single timestep as a dictionary.

        Note that because we have already checked which starting dates are valid in the
        `dates` property, we know that the requested dates will be consecutive in each
        data input, so we can use `get_tchw_slice` rather than `get_tchw`.

        Returns:
            A dictionary with dataset names as keys and a numpy array as the value.
            The shape of each array is:
            - input datasets: [n_history_steps, C_input_k, H_input_k, W_input_k]
            - target dataset: [n_forecast_steps, C_target, H_target, W_target]
            - target_uncertainty, when configured: same shape as the target

        """
        start_date = self.dates[idx]
        target_start = start_date + self.n_history_steps * self.frequency
        batch = {
            ds.name: ds.get_tchw_slice(start_date, self.n_history_steps, check=False)
            for ds in self.inputs
        } | {
            "target": self.target.get_tchw_slice(
                target_start,
                self.n_forecast_steps,
                check=False,
            )
        }
        if self.target_uncertainty is not None:
            batch["target_uncertainty"] = self.target_uncertainty.get_tchw_slice(
                target_start,
                self.n_forecast_steps,
                check=False,
            )
        return batch

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
