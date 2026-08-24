"""Uncertainty-aware visualisations for model evaluation."""

from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL.ImageFile import ImageFile

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import ArrayHW, PlotSpec

from .convert import image_from_figure
from .land_mask import LandMask
from .layout import (
    build_single_panel_figure,
    format_symmetric_ticks,
    set_suptitle_with_box,
)
from .plotting_core import colourmap_with_bad, create_normalisation

_SPATIAL_NDIM = 2


def compute_standardised_difference(
    ground_truth: ArrayHW,
    prediction: ArrayHW,
    uncertainty: ArrayHW,
) -> ArrayHW:
    """Return prediction error in units of observational standard uncertainty.

    The signed convention matches the existing plotting difference convention:
    ``ground_truth - prediction``. Locations with non-finite or non-positive
    uncertainty are returned as NaN because a z value is undefined there.

    Args:
        ground_truth: Two-dimensional observed values.
        prediction: Two-dimensional predicted values.
        uncertainty: Two-dimensional standard uncertainty for the observations.

    Returns:
        Two-dimensional standardised difference array.

    Raises:
        InvalidArrayError: If inputs are not two-dimensional arrays of equal shape.

    """
    arrays = (ground_truth, prediction, uncertainty)
    if any(array.ndim != _SPATIAL_NDIM for array in arrays):
        shapes = tuple(array.shape for array in arrays)
        msg = f"Expected 2D ground truth, prediction and uncertainty arrays, got {shapes}."
        raise InvalidArrayError(msg)
    if not (ground_truth.shape == prediction.shape == uncertainty.shape):
        msg = (
            "Ground truth, prediction and uncertainty must have matching shapes; "
            f"got {ground_truth.shape}, {prediction.shape} and {uncertainty.shape}."
        )
        raise InvalidArrayError(msg)

    result = np.full(ground_truth.shape, np.nan, dtype=float)
    valid = np.isfinite(uncertainty) & (uncertainty > 0)
    np.divide(
        ground_truth - prediction,
        uncertainty,
        out=result,
        where=valid,
    )
    return result


def plot_static_uncertainty(  # noqa: PLR0913
    ground_truth: ArrayHW,
    prediction: ArrayHW,
    uncertainty: ArrayHW,
    *,
    date: date | datetime,
    land_mask: LandMask,
    plot_spec: PlotSpec,
    variable_name: str,
) -> dict[str, list[ImageFile]]:
    """Plot signed prediction error in units of observational uncertainty.

    A value of ``z=1`` means the observation exceeds the prediction by one
    reported standard uncertainty. Invalid or non-positive uncertainty values
    are masked.
    """
    z_difference = compute_standardised_difference(
        ground_truth, prediction, uncertainty
    )
    z_difference = land_mask.apply_to(z_difference)

    height, width = z_difference.shape
    fig, ax, cax = build_single_panel_figure(
        height=height,
        width=width,
        colourbar_location=plot_spec.colourbar_location,
    )

    norm, vmin, vmax = create_normalisation(z_difference, centre=0.0)
    image = ax.imshow(
        z_difference,
        cmap=colourmap_with_bad("RdBu_r", bad_color="lightgrey"),
        norm=norm,
        origin="lower",
        interpolation="nearest",
    )

    is_vertical = plot_spec.colourbar_location == "vertical"
    colourbar = fig.colorbar(
        image,
        ax=ax,
        cax=cax,
        orientation=plot_spec.colourbar_location,
    )
    format_symmetric_ticks(
        colourbar,
        vmin=vmin,
        vmax=vmax,
        decimals=1,
        is_vertical=is_vertical,
        centre=0.0,
        use_scientific_notation=False,
    )

    shown = date.date().isoformat() if isinstance(date, datetime) else date.isoformat()
    set_suptitle_with_box(
        fig,
        f"{variable_name} standardised difference (z)   Shown: {shown}",
    )

    try:
        image_file = image_from_figure(fig, dpi=plot_spec.dpi)
    finally:
        plt.close(fig)

    return {f"{shown}-{variable_name}-uncertainty-z": [image_file]}
