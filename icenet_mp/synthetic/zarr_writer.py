"""Write synthetic frame sequences as anemoi-compatible zarr stores.

Mirrors the minimal zarr layout that ``anemoi.datasets.open_dataset`` expects (as used
by ``tests/conftest.py``'s ``build_zarr``), so a synthetic sequence can be consumed
through the exact same ``SingleDataset``/``CombinedDataset``/``CommonDataModule``
pipeline that real datasets use, without going through the anemoi ``Create()`` pipeline.
"""

import datetime
from pathlib import Path

import numpy as np
import zarr

from icenet_mp.types import ArrayTHW


def daily_dates(
    n_timesteps: int,
    start_date: datetime.datetime = datetime.datetime(2020, 1, 1),  # noqa: B008
) -> list[datetime.datetime]:
    """Return `n_timesteps` consecutive daily dates starting at `start_date`."""
    return [start_date + datetime.timedelta(days=step) for step in range(n_timesteps)]


def write_synthetic_zarr(
    zarr_path: Path,
    *,
    frames: ArrayTHW,
    variable_name: str = "ice_conc",
    start_date: datetime.datetime = datetime.datetime(2020, 1, 1),  # noqa: B008
) -> Path:
    """Write a single-variable [T, H, W] frame sequence as an anemoi-compatible zarr store."""
    n_timesteps, height, width = frames.shape
    dates = daily_dates(n_timesteps, start_date)

    # Anemoi's on-disk layout is [time, channels, ensemble, position]
    data = frames.reshape(n_timesteps, 1, 1, height * width).astype(np.float32)

    lat_grid, lon_grid = np.meshgrid(
        np.linspace(-90, 90, height), np.linspace(-180, 180, width), indexing="ij"
    )

    mean = data.mean(axis=(0, 2, 3)).astype(np.float64)
    stdev = data.std(axis=(0, 2, 3)).astype(np.float64)
    minimum = data.min(axis=(0, 2, 3)).astype(np.float64)
    maximum = data.max(axis=(0, 2, 3)).astype(np.float64)

    zarr_path.mkdir(parents=True, exist_ok=True)
    z = zarr.open_group(str(zarr_path), mode="w")
    z.create_dataset("data", data=data, chunks=(1, 1, 1, height * width))
    z.create_dataset(
        "dates",
        data=np.array([np.datetime64(d, "s") for d in dates], dtype="datetime64[s]"),
    )
    z.create_dataset("latitudes", data=lat_grid.ravel().astype(np.float64))
    z.create_dataset("longitudes", data=lon_grid.ravel().astype(np.float64))
    z.create_dataset("mean", data=mean)
    z.create_dataset("stdev", data=stdev)
    z.create_dataset("minimum", data=minimum)
    z.create_dataset("maximum", data=maximum)
    z.attrs.update(
        {
            "field_shape": [height, width],
            "frequency": "24h",
            "variables": [variable_name],
            "missing_dates": [],
            "flatten_grid": True,
            "ensemble_dimension": 2,
        }
    )
    return zarr_path
