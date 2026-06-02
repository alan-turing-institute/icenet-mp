import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from omegaconf import DictConfig


def build_zarr(
    zarr_path: Path,
    data_dict: dict[str, Any],
    full_dates: list[datetime.datetime] | None = None,
    missing_date_strs: list[str] | None = None,
) -> Path:
    """Write a minimal anemoi-compatible zarr store without using Create().

    Bypasses the slow anemoi create pipeline (which has ~40s fixed overhead) by writing
    the zarr arrays directly to disk.
    """
    data_dates: list[datetime.datetime] = data_dict["coords"]["time"]["data"]
    full_dates = full_dates if full_dates is not None else data_dates
    variables = list(data_dict["data_vars"].keys())
    lats: list[float] = data_dict["coords"]["lat"]["data"]
    lons: list[float] = data_dict["coords"]["lon"]["data"]

    T, C, H, W = len(full_dates), len(variables), len(lats), len(lons)  # noqa: N806

    # Build (T, C, 1, H * W) data array; missing timesteps stay zero.
    available_dates = {d.date() for d in data_dates}
    data = np.zeros((T, C, 1, H * W), dtype=np.float32)
    data_idx = 0
    for t, d in enumerate(full_dates):
        if d.date() in available_dates:
            for c, var in enumerate(variables):
                data[t, c, 0, :] = np.array(
                    data_dict["data_vars"][var]["data"][data_idx], dtype=np.float32
                ).ravel()
            data_idx += 1

    # Compute per-variable statistics from available timesteps only.
    flat = data[[d.date() in available_dates for d in full_dates], :, 0, :]
    means = flat.mean(axis=(0, 2)).astype(np.float64)
    stdevs = flat.std(axis=(0, 2)).astype(np.float64)
    minimums = flat.min(axis=(0, 2)).astype(np.float64)
    maximums = flat.max(axis=(0, 2)).astype(np.float64)

    # Create lat/lon grid
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    latitudes = lat_grid.ravel().astype(np.float64)
    longitudes = lon_grid.ravel().astype(np.float64)

    # Convert dates into timestamps at noon
    date_timestamps = np.array(
        [np.datetime64(f"{d.date()}T12:00:00", "s") for d in full_dates],
        dtype="datetime64[s]",
    )

    zarr_path.mkdir(parents=True, exist_ok=True)
    z = zarr.open_group(str(zarr_path), mode="w")
    z.create_dataset("data", data=data, chunks=(1, C, 1, H * W))
    z.create_dataset("dates", data=date_timestamps)
    z.create_dataset("latitudes", data=latitudes)
    z.create_dataset("longitudes", data=longitudes)
    z.create_dataset("mean", data=means)
    z.create_dataset("stdev", data=stdevs)
    z.create_dataset("minimum", data=minimums)
    z.create_dataset("maximum", data=maximums)
    z.attrs.update(
        {
            "field_shape": [H, W],
            "frequency": "24h",
            "variables": variables,
            "missing_dates": [f"{s[:10]}T12:00:00" for s in (missing_date_strs or [])],
            "flatten_grid": True,
            "ensemble_dimension": 2,
        }
    )
    return zarr_path


@pytest.fixture
def cfg_common_data_module() -> DictConfig:
    """Test configuration for a CommonDataModule."""
    return DictConfig(
        {
            "base_path": "/mock/base/path",
            "data": {
                "datasets": {"ds1": {"name": "mock", "group_as": "group1"}},
                "split": {
                    "batch_size": 2,
                    "predict": [{"start": None, "end": None}],
                    "test": [{"start": "2020-01-01", "end": "2020-12-31"}],
                    "train": [
                        {"start": None, "end": "2019-12-31"},
                        {"start": "2018-01-01", "end": None},
                    ],
                    "validate": [{"start": "2020-01-01", "end": "2020-03-31"}],
                },
            },
            "predict": {
                "target": {"group_name": "group1"},
                "n_forecast_steps": 1,
                "n_history_steps": 1,
            },
        }
    )


@pytest.fixture
def cfg_decoder() -> DictConfig:
    """Test configuration for a decoder."""
    return DictConfig({"_target_": "icenet_mp.models.decoders.NaiveLinearDecoder"})


@pytest.fixture
def cfg_encoders() -> DictConfig:
    """Test configuration for an encoder."""
    return DictConfig(
        {
            "latent_space": (64, 64),
            "test-input": {
                "_target_": "icenet_mp.models.encoders.NaiveLinearEncoder",
            },
        }
    )


@pytest.fixture
def cfg_input_space() -> DictConfig:
    """Test configuration for an input space."""
    return DictConfig(
        {
            "channels": 4,
            "name": "test-input",
            "shape": (512, 512),
        }
    )


@pytest.fixture
def cfg_model_service() -> DictConfig:
    """Test configuration for a ModelService."""
    return DictConfig(
        {
            "data": {
                "datasets": {
                    "mock-dataset-1": {
                        "name": "mock_dataset",
                        "group_as": "mock-dataset-group-1",
                    },
                    "mock-dataset-2": {
                        "name": "mock_dataset",
                        "group_as": "mock-dataset-group-2",
                    },
                },
                "split": {
                    "batch_size": 2,
                    "predict": [{"start": None, "end": None}],
                    "test": [{"start": "2019-01-01", "end": "2019-01-31"}],
                    "train": [
                        {"start": "2017-01-01", "end": "2017-12-31"},
                        {"start": "2018-02-01", "end": "2018-12-31"},
                    ],
                    "validate": [{"start": "2018-01-01", "end": "2018-01-31"}],
                },
            },
            "evaluate": {"callbacks": {}},
            "hemisphere": "north",
            "loggers": {},
            "model": {
                "_target_": "MockModel",
                "name": "mock-model",
            },
            "predict": {
                "target": {"group_name": "mock-dataset-group-1"},
                "n_forecast_steps": 2,
                "n_history_steps": 3,
            },
            "train": {
                "callbacks": {},
                "optimizer": {},
                "scheduler": {},
                "trainer": {},
            },
        }
    )


@pytest.fixture
def cfg_optimizer() -> DictConfig:
    """Test configuration for an optimizer."""
    return DictConfig({"_target_": "torch.optim.AdamW", "lr": 5e-4})


@pytest.fixture
def cfg_output_space() -> DictConfig:
    """Test configuration for an output space."""
    return DictConfig(
        {
            "channels": 1,
            "name": "target",
            "shape": (432, 432),
        }
    )


@pytest.fixture
def cfg_processor() -> DictConfig:
    """Test configuration for a processor."""
    return DictConfig({"_target_": "icenet_mp.models.processors.NullProcessor"})


@pytest.fixture
def cfg_scheduler() -> DictConfig:
    """Test configuration for a scheduler."""
    return DictConfig(
        {
            "_target_": "torch.optim.lr_scheduler.LinearLR",
            "lr_scheduler_parameters": {"frequency": 1, "interval": "epoch"},
            "scheduler_parameters": {"start_factor": 0.2, "end_factor": 0.8},
        }
    )


@pytest.fixture(scope="session")
def mock_data() -> dict[str, dict[str, Any]]:
    """Fixture to create a mock dataset for testing."""
    return {
        "coords": {
            "lat": {
                "dims": ("lat"),
                "attrs": {"units": "degrees_north", "standard_name": "latitude"},
                "data": [-89, -90],
            },
            "lon": {
                "dims": ("lon"),
                "attrs": {"units": "degrees_east", "standard_name": "longitude"},
                "data": [44, 45],
            },
            "time": {
                "dims": ("time",),
                "attrs": {"standard_name": "time"},
                "data": [
                    datetime.datetime(2020, 1, 1, 0, 0, 0),
                    datetime.datetime(2020, 1, 2, 0, 0, 0),
                    datetime.datetime(2020, 1, 3, 0, 0, 0),
                    datetime.datetime(2020, 1, 4, 0, 0, 0),
                    datetime.datetime(2020, 1, 5, 0, 0, 0),
                ],
            },
        },
        "attrs": {},
        "dims": {"lat": 2, "lon": 2, "time": 5},
        "data_vars": {
            "ice_conc": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [
                    [[0.5, 1.0], [0.4, 0.0]],
                    [[0.4, 0.9], [0.3, 0.1]],
                    [[0.3, 0.8], [0.2, 0.2]],
                    [[0.2, 0.7], [0.1, 0.3]],
                    [[0.1, 0.6], [0.0, 0.4]],
                ],
            },
            "ice_thickness": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [
                    [[1.5, 1.0], [1.6, 0.0]],
                    [[1.5, 0.9], [1.6, 0.1]],
                    [[1.5, 0.9], [1.6, 0.1]],
                    [[1.5, 0.9], [1.6, 0.2]],
                    [[1.5, 0.8], [1.6, 0.2]],
                ],
            },
            "temperature": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [
                    [[273.0, 274.0], [275.0, 276.0]],
                    [[273.5, 274.5], [275.5, 276.5]],
                    [[274.0, 275.0], [276.0, 277.0]],
                    [[274.5, 275.5], [276.5, 277.5]],
                    [[275.0, 276.0], [277.0, 278.0]],
                ],
            },
        },
    }


@pytest.fixture(scope="session")
def mock_data_missing_dates() -> dict[str, dict[str, Any]]:
    """Fixture to create a mock dataset with missing dates for testing."""
    return {
        "coords": {
            "lat": {
                "dims": ("lat"),
                "attrs": {"units": "degrees_north", "standard_name": "latitude"},
                "data": [-89, -90],
            },
            "lon": {
                "dims": ("lon"),
                "attrs": {"units": "degrees_east", "standard_name": "longitude"},
                "data": [44, 45],
            },
            "time": {
                "dims": ("time",),
                "attrs": {"standard_name": "time"},
                "data": [
                    datetime.datetime(2020, 1, 1, 0, 0, 0),
                    datetime.datetime(2020, 1, 3, 0, 0, 0),
                    datetime.datetime(2020, 1, 5, 0, 0, 0),
                ],
            },
        },
        "attrs": {},
        "dims": {"lat": 2, "lon": 2, "time": 3},
        "data_vars": {
            "ice_conc": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [
                    [[0.5, 1.0], [0.4, 0.0]],
                    [[0.3, 0.8], [0.2, 0.2]],
                    [[0.1, 0.6], [0.0, 0.4]],
                ],
            }
        },
    }


@pytest.fixture(scope="session")
def mock_data_non_normalized_times(
    mock_data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fixture to create a mock dataset for testing."""
    output = dict(**mock_data)
    output["coords"]["time"]["data"] = [
        datetime.datetime(2020, 1, 1, 3, 47, 42),
        datetime.datetime(2020, 1, 2, 3, 47, 42),
        datetime.datetime(2020, 1, 3, 3, 47, 42),
        datetime.datetime(2020, 1, 4, 3, 47, 42),
        datetime.datetime(2020, 1, 5, 3, 47, 42),
    ]
    return output


@pytest.fixture(scope="session")
def mock_data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture to create a temporary directory for mock data files."""
    return tmp_path_factory.mktemp("data", numbered=False)


@pytest.fixture(scope="session")
def mock_dataset(mock_data_path: Path, mock_data: dict[str, dict[str, Any]]) -> Path:
    """Fixture to create a mock file for testing."""
    return build_zarr(mock_data_path / "anemoi" / "mock_dataset.zarr", mock_data)


@pytest.fixture(scope="session")
def mock_dataset_missing_dates(
    mock_data_path: Path,
    mock_data_missing_dates: dict[str, dict[str, Any]],
) -> Path:
    """Fixture to create a mock file with missing dates for testing."""
    return build_zarr(
        mock_data_path / "anemoi" / "mock_dataset_missing_dates.zarr",
        mock_data_missing_dates,
        full_dates=[
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2020, 1, 2),
            datetime.datetime(2020, 1, 3),
            datetime.datetime(2020, 1, 4),
            datetime.datetime(2020, 1, 5),
        ],
        missing_date_strs=["2020-01-02", "2020-01-04"],
    )


@pytest.fixture(scope="session")
def mock_dataset_non_normalized_times(
    mock_data_path: Path,
    mock_data_non_normalized_times: dict[str, dict[str, Any]],
) -> Path:
    """Fixture to create a mock file with non-normalized times for testing."""
    return build_zarr(
        mock_data_path / "anemoi" / "mock_dataset_non_normalized_times.zarr",
        mock_data_non_normalized_times,
    )


def make_varying_sic_stream(
    *,
    dist_grid: np.ndarray,
    timesteps: int,
    base_radius: float,
    rng: np.random.Generator,
    ring_width: float = 6.0,
    noise_std: float = 0.03,
    radius_oscillation_amplitude: float = 0.5,
    radius_oscillation_frequency: float = 0.7,
) -> np.ndarray:
    """Vectorized [T, H, W] sea-ice concentration with oscillating coastline radius.

    - Coastline radius varies sinusoidally over time about base_radius
    - Concentration decays exponentially with distance outside the coastline
    - Adds small Gaussian noise and masks land (inside radius) to 0.0
    """
    height, width = dist_grid.shape

    t_idx = np.arange(timesteps, dtype=float)
    radius_t = base_radius + radius_oscillation_amplitude * np.sin(
        radius_oscillation_frequency * t_idx
    )  # [T]

    dist_b = dist_grid[None, :, :]  # [1,H,W]
    radius_b = radius_t[:, None, None]  # [T,1,1]

    outside = np.maximum(0.0, dist_b - radius_b)  # [T,H,W]
    falloff = np.exp(-(outside / max(1.0, float(ring_width))))

    noise = rng.normal(0.0, noise_std, size=(timesteps, height, width))
    sic = np.clip(falloff + noise, 0.0, 1.0)

    # Land mask: inside coastline set to 0.0 (temporary until NaN land masking)
    sic[dist_b < radius_b] = 0.0
    return sic.astype(np.float32)
