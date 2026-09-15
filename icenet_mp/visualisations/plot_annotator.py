import logging
from collections.abc import Sequence
from datetime import date, datetime

from icenet_mp.types import Metadata, PlotSpec

logger = logging.getLogger(__name__)


class PlotAnnotator:
    """Compose and draw titles, footers and warning badges for use in media."""

    def __init__(self, metadata: Metadata, plot_spec: PlotSpec) -> None:
        """Bind the metadata and plot spec shared by every title/footer."""
        self.plot_spec = plot_spec
        self.metadata = metadata

    def footer_for_static(self) -> str:
        """Build footer text for static plots using metadata that used to be in title."""
        lines: list[str] = []
        if subtitle := self.format_subtitle():
            lines.append(subtitle)
        return "\n".join(lines)

    def footer_for_video(self, dates: Sequence[date | datetime]) -> str:
        """Build footer text for video plots: animation range and metadata."""
        lines: list[str] = []
        if dates:
            start_s = self.format_date_for_title(dates[0])
            end_s = self.format_date_for_title(dates[-1])
            lines.append(f"Animating from {start_s} to {end_s}")
        if subtitle := self.format_subtitle():
            lines.append(subtitle)
        return "\n".join(lines)

    def format_date_for_title(self, dt: date | datetime) -> str:
        """Format a date/datetime to ISO date string (YYYY-MM-DD) for plot titles.

        Args:
            dt: Date or datetime object to format.

        Returns:
            ISO format date string (YYYY-MM-DD). Time components are stripped
            from datetime objects.

        Example:
            >>> from datetime import date, datetime
            >>> from icenet_mp.types import Metadata, PlotSpec
            >>> annotator = PlotAnnotator(Metadata(), PlotSpec())
            >>> annotator.format_date_for_title(date(2023, 12, 25))
            '2023-12-25'
            >>> annotator.format_date_for_title(datetime(2023, 12, 25, 14, 30))
            '2023-12-25'

        """
        if isinstance(dt, datetime):
            return dt.date().isoformat()
        return dt.isoformat()

    def format_subtitle(self) -> str | None:  # noqa: C901, PLR0912
        """Format the bound metadata as a compact multi-line subtitle for plot titles.

        Lines:
          1) Model: <model>  Epoch: <num>  Training Dates: <start> — <end> (<cadence>) <num>pts
          2) Training Data: <source> (<vars>) <source> (<vars>)

        Returns:
            Formatted metadata string with newlines, or None if no metadata available.

        """
        lines: list[str] = []

        # Line 1: Model/Epoch/Dates
        info_parts: list[str] = []
        if self.metadata.model:
            info_parts.append(f"Model: {self.metadata.model}")
        if self.metadata.current_epoch is not None:
            info_parts.append(f"Epoch: {self.metadata.current_epoch}")

        if self.metadata.start or self.metadata.end:
            dates_part = f"Training Dates: {self.metadata.start or '?'} — {self.metadata.end or '?'}"
            if self.metadata.cadence:
                if (
                    self.metadata.n_history_steps is not None
                    and self.metadata.n_history_steps > 0
                ):
                    dates_part += (
                        f" ({self.metadata.cadence}, "
                        f"{self.metadata.n_history_steps} step history)"
                    )
                else:
                    dates_part += f" ({self.metadata.cadence})"
            if self.metadata.n_points is not None:
                dates_part += f" {self.metadata.n_points} pts"
            info_parts.append(dates_part)

        if info_parts:
            lines.append("  ".join(info_parts))

        # Line 2: Training data sources and variables
        if self.metadata.vars_by_source:
            source_parts = []
            for source in sorted(self.metadata.vars_by_source.keys()):
                vars_list = self.metadata.vars_by_source[source]
                if vars_list:
                    vars_str = ",".join(vars_list)
                    source_parts.append(f"{source} ({vars_str})")
                else:
                    source_parts.append(source)
            if source_parts:
                lines.append(f"Training Data: {' '.join(source_parts)}")

        return "\n".join(lines) if lines else None

    def format_title(
        self,
        variable: str,
        when: date | datetime,
        units: str | None,
    ) -> str:
        """Format a title string for a raw input variable plot.

        Args:
            variable: Variable name.
            when: Date or datetime of the data.
            units: Display units for the variable.

        Returns:
            Formatted title string.

        """
        hemisphere = self.plot_spec.hemisphere
        hemi = f" ({hemisphere.capitalize()})" if hemisphere else ""
        units_s = f" [{units}]" if units else ""
        shown = (
            when.date().isoformat() if isinstance(when, datetime) else when.isoformat()
        )
        return f"{variable}{units_s}{hemi}   Shown: {shown}"

    def formatted_variable_name(self, variable: str) -> str:
        """Return a human-friendly variable name for titles.

        Example: "sea_ice_concentration" -> "Sea ice concentration".
        """
        pretty = variable.replace("_", " ").strip()
        return pretty.title() if pretty else ""

    def title_for_static(self, variable_name: str, when: date | datetime) -> str:
        """Compose a simple title for static plots.

        Lines:
          1) "<Variable> (<Hemisphere>)  Shown: YYYY-MM-DD"
             (Footer contains any metadata such as model/epoch/training data if present)
        """
        metric = self.formatted_variable_name(variable_name)
        hemisphere = self.plot_spec.hemisphere
        hemi = f" ({hemisphere.capitalize()})" if hemisphere else ""
        return f"{metric}{hemi} Prediction   Shown: {self.format_date_for_title(when)}"

    def title_for_video(
        self,
        variable_name: str,
        dates: Sequence[date | datetime],
        current_index: int,
    ) -> str:
        """Compose a simple title for video plots (date changes per frame).

        Lines:
          1) "<Variable> (<Hemisphere>)  Frame: YYYY-MM-DD"
          2) Footer: "Animating from <start> to <end>"
          3) Footer: "Model: <model>  Epoch: <num>  Training Dates: <start> — <end> (<cadence>) <num> pts" (optional)
          4) Footer: "Training Data: <source> (<vars>) <source> (<vars>)" (optional)
        """
        metric = self.formatted_variable_name(variable_name)
        hemisphere = self.plot_spec.hemisphere
        hemi = f" ({hemisphere.capitalize()})" if hemisphere else ""
        if dates:
            shown = self.format_date_for_title(dates[current_index])
            return f"{metric}{hemi} Prediction   Frame: {shown}"
        return f"{metric}{hemi} Prediction"
