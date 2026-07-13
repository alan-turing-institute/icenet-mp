from collections.abc import Sequence

import numpy as np
from earthkit.data import ArrayField


def make_array_field(
    values: np.ndarray,
    lats: Sequence[float],
    lons: Sequence[float],
    param: str = "siconc",
) -> ArrayField:
    """Build a real earthkit ArrayField with the given values and lat/lon grid."""
    return ArrayField(
        values,
        {"param": param, "latitudes": np.asarray(lats), "longitudes": np.asarray(lons)},
    )
