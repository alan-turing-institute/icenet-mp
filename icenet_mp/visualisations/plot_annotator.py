import logging
from collections.abc import Sequence
from datetime import date, datetime

from matplotlib.figure import Figure
from matplotlib.text import Text

from icenet_mp.types import PlotSpec

from .layout import draw_badge_with_box, set_footer_with_box

logger = logging.getLogger(__name__)


class PlotAnnotator:
    """Composes and draws titles, footers and warning badges for sea-ice plots."""

    def format_title(
        self,
        variable: str,
        hemisphere: str | None,
        when: date | datetime,
        units: str | None,
    ) -> str:
        """Format a title string for a raw input variable plot.

        Args:
            variable: Variable name.
            hemisphere: Hemisphere ("north" or "south"), if applicable.
            when: Date or datetime of the data.
            units: Display units for the variable.

        Returns:
            Formatted title string.

        """
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

    def format_date_for_title(self, dt: date | datetime) -> str:
        """Format a date/datetime to ISO date string (YYYY-MM-DD) for plot titles.

        Args:
            dt: Date or datetime object to format.

        Returns:
            ISO format date string (YYYY-MM-DD). Time components are stripped
            from datetime objects.

        Example:
            >>> from datetime import date, datetime
            >>> PlotAnnotator().format_date_for_title(date(2023, 12, 25))
            '2023-12-25'
            >>> PlotAnnotator().format_date_for_title(datetime(2023, 12, 25, 14, 30))
            '2023-12-25'

        """
        if isinstance(dt, datetime):
            return dt.date().isoformat()
        return dt.isoformat()

    def title_for_static(
        self, variable_name: str, plot_spec: PlotSpec, when: date | datetime
    ) -> str:
        """Compose a simple suptitle for static plots.

        Lines:
          1) "<Variable> (<Hemisphere>)  Shown: YYYY-MM-DD"
             (Footer contains any metadata such as model/epoch/training data if present)
        """
        metric = self.formatted_variable_name(variable_name)
        hemi = f" ({plot_spec.hemisphere.capitalize()})" if plot_spec.hemisphere else ""
        return f"{metric}{hemi} Prediction   Shown: {self.format_date_for_title(when)}"

    def title_for_video(
        self,
        variable_name: str,
        plot_spec: PlotSpec,
        dates: Sequence[date | datetime],
        current_index: int,
    ) -> str:
        """Compose a simple suptitle for video plots (date changes per frame).

        Lines:
          1) "<Variable> (<Hemisphere>)  Frame: YYYY-MM-DD"
          2) Footer: "Animating from <start> to <end>"
          3) Footer: "Model: <model>  Epoch: <num>  Training Dates: <start> — <end> (<cadence>) <num> pts" (optional)
          4) Footer: "Training Data: <source> (<vars>) <source> (<vars>)" (optional)
        """
        metric = self.formatted_variable_name(variable_name)
        hemi = f" ({plot_spec.hemisphere.capitalize()})" if plot_spec.hemisphere else ""
        if dates:
            shown = self.format_date_for_title(dates[current_index])
            return f"{metric}{hemi} Prediction   Frame: {shown}"
        return f"{metric}{hemi} Prediction"

    def footer_for_static(self, plot_spec: PlotSpec) -> str:
        """Build footer text for static plots using metadata that used to be in title."""
        lines: list[str] = []
        if plot_spec.metadata_subtitle:
            lines.append(plot_spec.metadata_subtitle)
        return "\n".join(lines)

    def footer_for_video(
        self, plot_spec: PlotSpec, dates: Sequence[date | datetime]
    ) -> str:
        """Build footer text for video plots: animation range and metadata."""
        lines: list[str] = []
        if dates:
            start_s = self.format_date_for_title(dates[0])
            end_s = self.format_date_for_title(dates[-1])
            lines.append(f"Animating from {start_s} to {end_s}")
        if plot_spec.metadata_subtitle:
            lines.append(plot_spec.metadata_subtitle)
        return "\n".join(lines)

    def warning_badge(
        self,
        fig: Figure,
        title_text: Text | None,
        warnings: Sequence[str],
    ) -> None:
        """Render a warning badge close to the title."""
        if not warnings:
            return
        badge = "Warnings: " + ", ".join(warnings)
        if title_text is not None:
            _, title_y = title_text.get_position()
            n_lines = title_text.get_text().count("\n") + 1
            warning_y = max(title_y - (0.05 + 0.02 * (n_lines - 1)), 0.0)
        else:
            warning_y = 0.90
        draw_badge_with_box(fig, 0.5, warning_y, badge)

    def maybe_add_footer(self, fig: Figure, plot_spec: PlotSpec) -> None:
        """Attach footer metadata when enabled."""
        if not getattr(plot_spec, "include_footer_metadata", True):
            return
        try:
            footer_text = self.footer_for_static(plot_spec)
            if footer_text:
                set_footer_with_box(fig, footer_text)
        except Exception:
            logger.exception("Failed to draw footer; continuing without footer.")
