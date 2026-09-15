import numpy as np

from icenet_mp.data import CombinedDataset
from icenet_mp.types import Metadata


class MetadataBuilder:
    """Builds Metadata from a dataset."""

    @staticmethod
    def _format_cadence(frequency: np.timedelta64) -> str:
        """Format a dataset's frequency as a short, human-readable cadence label."""
        hours = float(frequency / np.timedelta64(1, "h"))
        if hours % 24 == 0:
            days = int(hours // 24)
            return "daily" if days == 1 else f"{days}d"
        return "hourly" if hours == 1 else f"{hours:g}h"

    def from_dataset(
        self,
        dataset: CombinedDataset,
        *,
        current_epoch: int | None = None,
        model_name: str | None = None,
    ) -> Metadata:
        """Build structured metadata from a CombinedDataset's realised state.

        Uses the dataset's actual start/end dates, frequency, length and
        variable names, rather than parsing the raw Hydra config -- which was
        both less accurate (didn't account for missing dates) and, for
        cadence, broken (it read a config key no real config has, so cadence/
        n_points/n_history_steps never actually appeared on a subtitle).
        """
        vars_by_source = {ds.name: sorted(ds.variable_names) for ds in dataset.inputs}
        return Metadata(
            model=model_name or None,
            current_epoch=current_epoch,
            start=str(dataset.start_date.astype("datetime64[D]")),
            end=str(dataset.end_date.astype("datetime64[D]")),
            cadence=self._format_cadence(dataset.frequency),
            n_points=len(dataset),
            n_history_steps=dataset.n_history_steps,
            vars_by_source=vars_by_source or None,
        )
