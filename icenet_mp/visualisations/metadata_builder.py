"""Metadata extraction and formatting for plot titles."""

import numpy as np

from icenet_mp.data import CombinedDataset
from icenet_mp.types import Metadata


class MetadataBuilder:
    """Builds Metadata from a dataset's realised state and formats it for plot titles."""

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

    def format_subtitle(self, metadata: Metadata) -> str | None:  # noqa: C901, PLR0912
        """Format metadata dataclass as a compact multi-line subtitle for plot titles.

        Lines:
          1) Model: <model>  Epoch: <num>  Training Dates: <start> — <end> (<cadence>) <num>pts
          2) Training Data: <source> (<vars>) <source> (<vars>)

        Args:
            metadata: Metadata dataclass instance to format.

        Returns:
            Formatted metadata string with newlines, or None if no metadata available.

        """
        lines: list[str] = []

        # Line 1: Model/Epoch/Dates
        info_parts: list[str] = []
        if metadata.model:
            info_parts.append(f"Model: {metadata.model}")
        if metadata.current_epoch is not None:
            info_parts.append(f"Epoch: {metadata.current_epoch}")

        if metadata.start or metadata.end:
            dates_part = (
                f"Training Dates: {metadata.start or '?'} — {metadata.end or '?'}"
            )
            if metadata.cadence:
                if (
                    metadata.n_history_steps is not None
                    and metadata.n_history_steps > 0
                ):
                    dates_part += (
                        f" ({metadata.cadence}, "
                        f"{metadata.n_history_steps} step history)"
                    )
                else:
                    dates_part += f" ({metadata.cadence})"
            if metadata.n_points is not None:
                dates_part += f" {metadata.n_points} pts"
            info_parts.append(dates_part)

        if info_parts:
            lines.append("  ".join(info_parts))

        # Line 2: Training data sources and variables
        if metadata.vars_by_source:
            source_parts = []
            for source in sorted(metadata.vars_by_source.keys()):
                vars_list = metadata.vars_by_source[source]
                if vars_list:
                    vars_str = ",".join(vars_list)
                    source_parts.append(f"{source} ({vars_str})")
                else:
                    source_parts.append(source)
            if source_parts:
                lines.append(f"Training Data: {' '.join(source_parts)}")

        return "\n".join(lines) if lines else None
