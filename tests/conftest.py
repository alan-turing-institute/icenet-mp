import datetime
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
import xarray as xr
from anemoi.datasets.commands.create import Create
from omegaconf import DictConfig

from icenet_mp.types import AnemoiCreateArgs


@pytest.fixture
def cfg_decoder() -> DictConfig:
    """Test configuration for a decoder."""
    return DictConfig({"_target_": "icenet_mp.models.decoders.NaiveLinearDecoder"})


@pytest.fixture
def cfg_encoder() -> DictConfig:
    """Test configuration for an encoder."""
    return DictConfig(
        {
            "_target_": "icenet_mp.models.encoders.NaiveLinearEncoder",
            "latent_space": (64, 64),
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
            "frequency": 1,
            "interval": "epoch",
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
    # Use the mock data to create a NetCDF file
    netcdf_path = mock_data_path / "mock_dataset.nc"
    xr.Dataset.from_dict(mock_data).to_netcdf(netcdf_path)
    # Create an Anemoi dataset from the NetCDF file
    config = DictConfig(
        {
            "dates": {
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-05T23:00:00",
                "frequency": "24h",
            },
            "input": {
                "netcdf": {
                    "path": str(netcdf_path),
                }
            },
        }
    )
    zarr_path = mock_data_path / "anemoi" / "mock_dataset.zarr"
    Create().run(
        AnemoiCreateArgs(
            path=str(zarr_path),
            config=config,
            overwrite=True,
        )
    )
    return Path(str(zarr_path))


@pytest.fixture(scope="session")
def mock_dataset_missing_dates(
    mock_data_path: Path,
    mock_data_missing_dates: dict[str, dict[str, Any]],
) -> Path:
    """Fixture to create a mock file with missing dates for testing."""
    # Use the mock data to create a NetCDF file
    netcdf_path = mock_data_path / "mock_dataset_missing_dates.nc"
    xr.Dataset.from_dict(mock_data_missing_dates).to_netcdf(netcdf_path)
    # Create an Anemoi dataset from the NetCDF file
    config = DictConfig(
        {
            "dates": {
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-05T23:00:00",
                "frequency": "24h",
                "missing": ["2020-01-02T00:00:00", "2020-01-04T00:00:00"],
            },
            "input": {
                "netcdf": {
                    "path": str(netcdf_path),
                }
            },
        }
    )
    zarr_path = mock_data_path / "anemoi" / "mock_dataset_missing_dates.zarr"
    Create().run(
        AnemoiCreateArgs(
            path=str(zarr_path),
            config=config,
            overwrite=True,
        )
    )
    return Path(str(zarr_path))


@pytest.fixture(scope="session")
def mock_dataset_non_normalized_times(
    mock_data_path: Path,
    mock_data_non_normalized_times: dict[str, dict[str, Any]],
) -> Path:
    """Fixture to create a mock file with non-normalized times for testing."""
    # Use the mock data to create a NetCDF file
    netcdf_path = mock_data_path / "mock_dataset_non_normalized_times.nc"
    xr.Dataset.from_dict(mock_data_non_normalized_times).to_netcdf(netcdf_path)
    # Create an Anemoi dataset from the NetCDF file
    config = DictConfig(
        {
            "dates": {
                "start": "2020-01-01T03:47:42",
                "end": "2020-01-05T23:00:00",
                "frequency": "24h",
            },
            "input": {
                "netcdf": {
                    "path": str(netcdf_path),
                }
            },
        }
    )
    zarr_path = mock_data_path / "anemoi" / "mock_dataset_non_normalized_times.zarr"
    Create().run(
        AnemoiCreateArgs(
            path=str(zarr_path),
            config=config,
            overwrite=True,
        )
    )
    return Path(str(zarr_path))


class MakeCircularArctic(Protocol):
    def __call__(
        self,
        height: int,
        width: int,
        *,
        rng: np.random.Generator,
        ring_width: int = ...,
        noise: float = ...,
    ) -> np.ndarray: ...


class CircularArcticFactory:
    """Callable factory for generating circular Arctic SIC maps.

    Defaults are provided at construction, but can be overridden per-call.
    Satisfies the `MakeCircularArctic` protocol.
    """

    def __init__(self, ring_width: int = 6, noise: float = 0.05) -> None:
        """Initialise the factory with default parameters.

        Args:
            ring_width: Width of the ring.
            noise: Noise level.

        """
        self.ring_width = ring_width
        self.noise = noise

    def __call__(
        self,
        height: int,
        width: int,
        *,
        rng: np.random.Generator,
        ring_width: int | None = None,
        noise: float | None = None,
    ) -> np.ndarray:
        """Generate a circular Arctic SIC map.

        Args:
            height: Height of the map.
            width: Width of the map.
            rng: Random number generator.
            ring_width: Width of the ring.
            noise: Noise level.

        """
        # Resolve per-call overrides or fall back to defaults
        effective_ring_width = self.ring_width if ring_width is None else ring_width
        effective_noise = self.noise if noise is None else noise

        # Create a grid of distances from the centre
        cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        # Choose a radius so the land takes most of the centre
        radius = min(height, width) * 0.25

        # Distance from the coastline (ring at the circle): 0 on the ring, >0 outside
        d_outside = np.maximum(0.0, dist - radius)

        # Ice strength: 1 at the ring, then smoothly falls to 0 with distance
        falloff = np.exp(-(d_outside / max(1.0, float(effective_ring_width))))

        # Add a tiny bit of texture so the ring looks less perfect
        texture = rng.normal(0.0, effective_noise, size=(height, width))
        sic = np.clip(falloff + texture, 0.0, 1.0)

        # Land mask: everything strictly inside the circle is NaN (temporary: set to 0)
        # Note: When land masking is implemented, set to np.nan instead of 0.0
        sic[dist < radius] = 0.0
        return sic.astype(np.float32)


@pytest.fixture
def make_circular_arctic() -> MakeCircularArctic:
    """Return a callable that creates a simple circular Arctic SIC map.

    Signature:
        (height: int, width: int, *, rng: np.random.Generator, ring_width: int = 6, noise: float = 0.05) -> np.ndarray
    """
    return CircularArcticFactory()


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


def _apply_scale(array: np.ndarray, scale: float) -> None:
    """In-place multiply array by scale (keeps dtype)."""
    array *= float(scale)


def _add_noise(
    array: np.ndarray, sigma: float, rng: np.random.Generator | None = None
) -> None:
    """In-place add Gaussian noise with std sigma. Deterministic if rng provided."""
    if rng is None:
        rng = np.random.default_rng(0)
    array += rng.normal(0.0, float(sigma), size=array.shape)


def _insert_outliers(array: np.ndarray, value: float, fraction: float = 0.1) -> None:
    """In-place set the first N flattened entries to `value` where N ~= fraction*size.

    Args:
        array: Array to insert outliers into.
        value: Value to insert.
        fraction: Fraction of entries to insert outliers into.

    """
    flat = array.ravel()
    n = int(max(1, round(float(fraction) * flat.size))) if fraction > 0 else 0
    if n > 0:
        flat[:n] = value
        array[:] = flat.reshape(array.shape)


def _make_bad_prediction(
    base_prediction: np.ndarray,
    *,
    scale: float | None = None,
    outlier: float | None = None,
    fraction: float = 0.0,
    noise: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a mutated copy of base_prediction according to the provided options.

    Args:
        base_prediction: Base prediction to mutate.
        scale: Scale to apply to the prediction.
        outlier: Outlier to insert into the prediction.
        fraction: Fraction of entries to insert outliers into.
        noise: Noise to add to the prediction.
        rng: Random number generator to use.

    Returns:
        Mutated prediction.

    """
    prediction = base_prediction.copy()
    if scale is not None:
        _apply_scale(prediction, scale)
    if noise is not None:
        _add_noise(prediction, noise, rng=rng)
    if outlier is not None and fraction > 0.0:
        _insert_outliers(prediction, outlier, fraction=fraction)
    return prediction


@pytest.fixture
def bad_prediction_maker() -> Callable[..., np.ndarray]:
    """Fixture returning a callable to produce mutated prediction arrays.

    Signature (positional):
        (base_prediction, *, scale=None, outlier=None, fraction=0.0, noise=None, rng=None) -> ndarray
    """
    return _make_bad_prediction
