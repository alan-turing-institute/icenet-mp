"""Time-series visualisations for forecast outputs."""

from collections.abc import Sequence
from datetime import date, datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import ArrayTHW

from .convert import image_from_figure
from .land_mask import LandMask


def _spatial_mean(values: np.ndarray) -> np.ndarray:
    """Return the finite-value spatial mean for every timestep."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=(1, 2))
    sums = np.where(finite, values, 0.0).sum(axis=(1, 2))
    return np.divide(
        sums,
        counts,
        out=np.full(values.shape[0], np.nan, dtype=float),
        where=counts > 0,
    )


def plot_time_trace(
    ground_truth: ArrayTHW,
    prediction: ArrayTHW,
    *,
    dates: Sequence[date | datetime],
    land_mask: LandMask,
    variable_name: str,
    dpi: int = 300,
) -> dict[str, list[ImageFile]]:
    """Plot spatial-mean ground truth and prediction values over forecast time.

    Args:
        ground_truth: Ground-truth array with shape [T, H, W].
        prediction: Prediction array with shape [T, H, W].
        dates: One date for each forecast timestep.
        land_mask: Land mask applied before spatial averaging.
        variable_name: Variable name used in the title and output key.
        dpi: Figure rendering resolution.

    Returns:
        A mapping containing a single rendered time-trace image.

    Raises:
        InvalidArrayError: If the arrays do not have matching THW shapes or the date
            count does not match the time dimension.

    """
    if ground_truth.ndim != 3 or prediction.ndim != 3:  # noqa: PLR2004
        msg = (
            "Time-trace inputs must have shape [T, H, W], got "
            f"{ground_truth.shape} and {prediction.shape}."
        )
        raise InvalidArrayError(msg)
    if ground_truth.shape != prediction.shape:
        msg = (
            f"Prediction ({prediction.shape}) has a different shape to ground truth "
            f"({ground_truth.shape})."
        )
        raise InvalidArrayError(msg)
    if len(dates) != ground_truth.shape[0]:
        msg = (
            f"Received {len(dates)} dates for {ground_truth.shape[0]} forecast "
            "timesteps."
        )
        raise InvalidArrayError(msg)

    ground_truth_mean = _spatial_mean(land_mask.apply_to(ground_truth))
    prediction_mean = _spatial_mean(land_mask.apply_to(prediction))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dates, ground_truth_mean, marker="o", label="Ground Truth")  # type: ignore[arg-type]
    ax.plot(dates, prediction_mean, marker="o", label="Prediction")  # type: ignore[arg-type]
    ax.set_xlabel("Forecast date")
    ax.set_ylabel(f"Spatial mean {variable_name}")
    ax.set_title(f"{variable_name} forecast trace")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_xticks(list(dates))  # type: ignore[arg-type]
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()

    try:
        return {f"{variable_name}-time-trace": [image_from_figure(fig, dpi=dpi)]}
    finally:
        plt.close(fig)
