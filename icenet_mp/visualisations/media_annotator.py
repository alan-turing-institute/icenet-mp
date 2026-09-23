import logging
from datetime import date, datetime, timedelta

from icenet_mp.types import Metadata, PlotSpec, Timespan
from icenet_mp.utils import iso_from_date

logger = logging.getLogger(__name__)


class MediaAnnotator:
    """Compose and draw titles, footers and warning badges for use in media."""

    def __init__(self, metadata: Metadata, plot_spec: PlotSpec) -> None:
        """Bind the metadata and plot spec shared by every title/footer."""
        self.plot_spec = plot_spec
        self.metadata = metadata

    @staticmethod
    def _format_variable_name(variable: str) -> str:
        """Return a human-friendly variable name for titles."""
        pretty = variable.replace("_", " ").strip()
        return pretty.title() if pretty else ""

    def _hemisphere_suffix(self) -> str:
        """Return ' (<Hemisphere>)' for the bound plot spec, or '' if unset."""
        hemisphere = self.plot_spec.hemisphere
        return f" ({hemisphere.capitalize()})" if hemisphere else ""

    def footer_for_static(self) -> str:
        """Build footer text for static plots using metadata that used to be in title."""
        lines: list[str] = []
        if subtitle := self.subtitle():
            lines.append(subtitle)
        return "\n".join(lines)

    def footer_for_video(self, forecast_ctx: Timespan) -> str:
        """Build footer text for video plots: animation range and metadata.

        Lines:
          1) "Animating from <start> to <end>"
          2) "Model: <model>  Epoch: <num>  Training Data: <start> — <end> (<cadence>) <num> pts" (optional)
          3) "Training Data: <source> (<vars>) <source> (<vars>)" (optional)
        """
        lines: list[str] = [
            f"Animating from {iso_from_date(forecast_ctx.start)} to {iso_from_date(forecast_ctx.end)}"
        ]
        if subtitle := self.subtitle():
            lines.append(subtitle)
        return "\n".join(lines)

    def subtitle(self) -> str | None:
        """Format the bound metadata as a compact multi-line subtitle for plot titles.

        Lines:
          1) Model: <model>  Epoch: <num>  Training Dates: <start> — <end> (<cadence>) <num>pts
          2) Training Data: <source> (<vars>) <source> (<vars>)

        Returns:
            Formatted metadata string with newlines, or None if no metadata available.

        """
        metadata = self.metadata
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
                    dates_part += f" ({metadata.cadence}, {metadata.n_history_steps} step history)"
                else:
                    dates_part += f" ({metadata.cadence})"
            if metadata.n_points is not None:
                dates_part += f" {metadata.n_points} pts"
            info_parts.append(dates_part)

        lines: list[str] = []
        if info_parts:
            lines.append("  ".join(info_parts))

        vars_by_source = metadata.vars_by_source
        if vars_by_source:
            source_parts = [
                f"{source} ({len(vars_by_source[source])} variables)"
                if vars_by_source[source]
                else source
                for source in sorted(vars_by_source)
            ]
            if source_parts:
                lines.append(f"Training Data: {' '.join(source_parts)}")

        return "\n".join(lines) if lines else None

    def title_for_static(
        self,
        *,
        forecast_date: datetime,
        history_ctx: Timespan,
        variable_name: str,
    ) -> str:
        """Compose a simple title for static plots.

        Args:
            history_ctx: The history context for the model.
            forecast_date: The date of the forecast.
            variable_name: Variable name.

        Returns:
            "<Variable> (<Hemisphere>) Input: YYYY-MM-DD - YYYY-MM-DD (<num steps> steps) Leadtime (+<leadtime> steps) YYYY-MM-DD"

        """
        metric = self._format_variable_name(variable_name)
        hemi_suffix = self._hemisphere_suffix()

        leadtime = (forecast_date - history_ctx.end).days

        return "  ".join(
            (
                f"{metric}{hemi_suffix}",
                f"History: {iso_from_date(history_ctx.start)} - {iso_from_date(history_ctx.end)} ({history_ctx.days} steps)",
                f"Leadtime (+{leadtime} steps): {iso_from_date(forecast_date)}",
            )
        )

    def title_for_variable(
        self,
        variable: str,
        when: date | datetime,
        units: str | None,
    ) -> str:
        """Format a title string for a standalone variable plot.

        Args:
            variable: Variable name.
            when: Date or datetime of the data.
            units: Display units for the variable.

        Returns:
            Formatted title string.

        """
        hemi_suffix = self._hemisphere_suffix()
        units_s = f" [{units}]" if units else ""
        return f"{variable}{units_s}{hemi_suffix} on {iso_from_date(when)}"

    def title_for_video(
        self,
        *,
        current_index: int,
        forecast_ctx: Timespan,
        history_ctx: Timespan,
        variable_name: str,
    ) -> str:
        """Compose a simple title for video plots (date changes per frame).

        Args:
            current_index: The index of the current frame.
            forecast_ctx: The forecast context for the model.
            history_ctx: The history context for the model.
            variable_name: Variable name.

        Returns:
            "<Variable> (<Hemisphere>) Input: YYYY-MM-DD - YYYY-MM-DD (<num steps>) Lead (+<leadtime> days) YYYY-MM-DD"

        """
        forecast_date = forecast_ctx.start + timedelta(days=current_index)
        return self.title_for_static(
            forecast_date=forecast_date,
            history_ctx=history_ctx,
            variable_name=variable_name,
        )
