"""Calendar-day-mean (climatology) table construction for a target dataset."""

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from icenet_mp.types import ArrayTCHW

from .calendar_day import (
    CALENDAR_DAY_LABELS,
    FEBRUARY_28_INDEX,
    FEBRUARY_29_INDEX,
    N_CALENDAR_DAYS,
    calendar_day_index,
)
from .single_dataset import SingleDataset

log = logging.getLogger(__name__)


def in_train_periods(
    day: np.datetime64, train_periods: Sequence[Mapping[str, str | None]]
) -> bool:
    """Return whether the date falls within any of the training period ranges.

    Bounds are compared at day precision, so a bound carrying a time component
    (e.g. ``2019-01-01T12:00:00``) behaves like ``2019-01-01``.
    """
    day_day = day.astype("datetime64[D]")
    for period in train_periods:
        start = period.get("start")
        end = period.get("end")
        if start is not None and day_day < np.datetime64(start).astype("datetime64[D]"):
            continue
        if end is not None and day_day > np.datetime64(end).astype("datetime64[D]"):
            continue
        return True
    return False


def build_climatology(
    target: SingleDataset, train_periods: Sequence[Mapping[str, str | None]]
) -> ArrayTCHW:
    """Return the climatology: calendar-day means of the target variables.

    The [366, C, H, W] table holds, for each calendar day (month/day label), the
    mean of the normalised target fields over dates sharing that calendar day
    within the averaging period. The averaging period is the union of the training
    split's date ranges, intersected with the dates available in the target
    dataset; it is never widened to dates outside the configured training periods.
    Dates that are missing from the dataset are never included in a mean.

    29 February is the exception: because a training period spanning only
    non-leap years has no such date, it is not required to have its own data. If
    no date in the averaging period falls on 29 February, that slot instead copies
    the 28 February mean.

    Raises:
        ValueError: If the training periods have no available dates at all, or a
            calendar day other than 29 February has no available dates in the
            period.

    """
    period_dates = [day for day in target.dates if in_train_periods(day, train_periods)]
    if not period_dates:
        msg = (
            "Cannot build climatology: none of the configured training periods "
            "have available dates in the target dataset "
            f"({target.start_date} to {target.end_date})."
        )
        raise ValueError(msg)
    by_day: dict[int, list[np.datetime64]] = defaultdict(list)
    for day in period_dates:
        by_day[calendar_day_index(day)].append(day)
    table = np.zeros((N_CALENDAR_DAYS, *target.space.chw), dtype=np.float64)
    for index, label in enumerate(CALENDAR_DAY_LABELS):
        day_dates = by_day.get(index, [])
        if not day_dates:
            if index == FEBRUARY_29_INDEX:
                log.info(
                    "Climatology: no 29 February dates in the averaging period; "
                    "using the 28 February mean for that day instead."
                )
                table[index] = table[FEBRUARY_28_INDEX]
                continue
            msg = (
                f"Cannot build climatology: calendar day {label} has no available "
                f"dates in the averaging period ({min(period_dates)} to "
                f"{max(period_dates)}). Check the configured training periods "
                "against the available data range."
            )
            raise ValueError(msg)
        table[index] = target.get_tchw(day_dates).astype(np.float64).mean(axis=0)
    log.info(
        "Climatology: computed calendar-day means over %d dates between %s and %s.",
        len(period_dates),
        min(period_dates),
        max(period_dates),
    )
    return table.astype(np.float32)
