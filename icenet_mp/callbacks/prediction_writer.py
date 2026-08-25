import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from netCDF4 import Dataset as NetCDFDataset
from torch import Tensor

from icenet_mp.data import CombinedDataset

logger = logging.getLogger(__name__)

_TIME_UNITS = "seconds since 1970-01-01 00:00:00"
_TIME_CALENDAR = "proleptic_gregorian"
_NTCHW_NDIM = 5


class PredictionWriter(Callback):
    """Write evaluation predictions to a CF-style NetCDF file."""

    def __init__(self, output_path: str | Path | None = None) -> None:
        """Configure the optional NetCDF output path."""
        super().__init__()
        self.output_path = None if output_path is None else Path(output_path)
        self._dataset: CombinedDataset | None = None
        self._file: Any | None = None
        self._sample_offset = 0

    @property
    def enabled(self) -> bool:
        """Return whether prediction export is enabled."""
        return self.output_path is not None

    @staticmethod
    def _load_dataset(trainer: Trainer) -> CombinedDataset:
        """Return the single CombinedDataset used for testing."""
        dataloaders: Any = trainer.test_dataloaders
        if isinstance(dataloaders, Sequence):
            if len(dataloaders) != 1:
                msg = "Prediction export supports exactly one test dataloader."
                raise ValueError(msg)
            dataloader = dataloaders[0]
        else:
            dataloader = dataloaders

        if dataloader is None or not isinstance(dataloader.dataset, CombinedDataset):
            msg = "Prediction export requires a CombinedDataset test dataloader."
            raise TypeError(msg)
        return dataloader.dataset

    @staticmethod
    def _seconds(values: np.ndarray) -> np.ndarray:
        """Convert datetime64 values to integer Unix seconds."""
        return values.astype("datetime64[s]").astype(np.int64)

    @staticmethod
    def _denormalise(
        prediction: np.ndarray,
        dataset: CombinedDataset,
    ) -> np.ndarray:
        """Convert model outputs from target normalisation back to source units."""
        minimum = np.asarray(dataset.target.statistics["minimum"], dtype=prediction.dtype)
        maximum = np.asarray(dataset.target.statistics["maximum"], dtype=prediction.dtype)
        if prediction.shape[2] != len(minimum) or len(minimum) != len(maximum):
            msg = (
                "Prediction channel count does not match target dataset statistics: "
                f"{prediction.shape[2]} channels vs {len(minimum)} statistics entries."
            )
            raise ValueError(msg)
        shape = (1, 1, -1, 1, 1)
        return prediction * (maximum - minimum).reshape(shape) + minimum.reshape(shape)

    def _initialise_file(self, dataset: CombinedDataset) -> None:
        """Create NetCDF dimensions, coordinates, and prediction variables."""
        if self.output_path is None:
            return

        height, width = dataset.target.space.shape
        latitudes = np.asarray(dataset.target.latitudes, dtype=np.float32)
        longitudes = np.asarray(dataset.target.longitudes, dtype=np.float32)
        if latitudes.size != height * width or longitudes.size != height * width:
            msg = (
                "Target latitude/longitude coordinate counts do not match the "
                f"target grid shape {dataset.target.space.shape}."
            )
            raise ValueError(msg)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = NetCDFDataset(str(self.output_path), "w", format="NETCDF4")
        self._file.setncattr("Conventions", "CF-1.10")
        self._file.setncattr("title", "IceNet-MP model predictions")
        self._file.setncattr("hemisphere", dataset.target.hemisphere)

        self._file.createDimension("forecast_reference_time", None)
        self._file.createDimension("lead_time", dataset.n_forecast_steps)
        self._file.createDimension("y", height)
        self._file.createDimension("x", width)

        reference_time = self._file.createVariable(
            "forecast_reference_time",
            "i8",
            ("forecast_reference_time",),
        )
        reference_time.standard_name = "forecast_reference_time"
        reference_time.units = _TIME_UNITS
        reference_time.calendar = _TIME_CALENDAR

        frequency_seconds = int(
            dataset.frequency.astype("timedelta64[s]").astype(np.int64)
        )
        lead_seconds = (
            np.arange(1, dataset.n_forecast_steps + 1, dtype=np.int64)
            * frequency_seconds
        )
        lead_time = self._file.createVariable("lead_time", "i8", ("lead_time",))
        lead_time.standard_name = "forecast_period"
        lead_time.long_name = "forecast lead time"
        lead_time.units = "seconds"
        lead_time[:] = lead_seconds

        valid_time = self._file.createVariable(
            "valid_time",
            "i8",
            ("forecast_reference_time", "lead_time"),
        )
        valid_time.standard_name = "time"
        valid_time.units = _TIME_UNITS
        valid_time.calendar = _TIME_CALENDAR

        latitude = self._file.createVariable("latitude", "f4", ("y", "x"))
        latitude.standard_name = "latitude"
        latitude.units = "degrees_north"
        latitude[:, :] = latitudes.reshape(height, width)

        longitude = self._file.createVariable("longitude", "f4", ("y", "x"))
        longitude.standard_name = "longitude"
        longitude.units = "degrees_east"
        longitude[:, :] = longitudes.reshape(height, width)

        for variable_name in dataset.target.variable_names:
            variable = self._file.createVariable(
                variable_name,
                "f4",
                ("forecast_reference_time", "lead_time", "y", "x"),
                zlib=True,
                complevel=4,
                fill_value=np.nan,
            )
            variable.coordinates = (
                "forecast_reference_time lead_time valid_time latitude longitude"
            )
            if variable_name == "ice_conc":
                variable.standard_name = "sea_ice_area_fraction"
                variable.long_name = "sea ice concentration"
                variable.units = "1"

    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Initialise prediction export before evaluation batches run."""
        if not self.enabled:
            return
        if getattr(trainer, "world_size", 1) != 1:
            msg = (
                "NetCDF prediction export currently requires single-process evaluation. "
                "Set evaluate.trainer.devices=1 when using --save-predictions."
            )
            raise RuntimeError(msg)

        self._dataset = self._load_dataset(trainer)
        self._sample_offset = 0
        self._initialise_file(self._dataset)

    def on_test_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ANN401, ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Append one evaluation batch to the NetCDF file."""
        if not self.enabled:
            return
        if self._dataset is None or self._file is None:
            msg = "Prediction writer was not initialised before receiving a test batch."
            raise RuntimeError(msg)
        if not isinstance(outputs, Mapping) or not isinstance(
            prediction_tensor := outputs.get("prediction"),
            Tensor,
        ):
            msg = "Prediction writer expected test outputs containing a prediction tensor."
            raise TypeError(msg)

        prediction = prediction_tensor.detach().float().cpu().numpy()
        if prediction.ndim != _NTCHW_NDIM:
            msg = (
                "Prediction writer expected NTCHW predictions, "
                f"received shape {prediction.shape}."
            )
            raise ValueError(msg)
        if prediction.shape[1] != self._dataset.n_forecast_steps:
            msg = (
                "Prediction forecast-step count does not match the test dataset: "
                f"{prediction.shape[1]} vs {self._dataset.n_forecast_steps}."
            )
            raise ValueError(msg)

        prediction = self._denormalise(prediction, self._dataset)
        batch_size = prediction.shape[0]
        start = self._sample_offset
        end = start + batch_size
        start_dates = np.asarray(self._dataset.dates[start:end])
        if len(start_dates) != batch_size:
            msg = (
                "Prediction writer received more samples than are available in the "
                "test dataset."
            )
            raise IndexError(msg)

        reference_dates = start_dates + (
            self._dataset.n_history_steps - 1
        ) * self._dataset.frequency
        reference_seconds = self._seconds(reference_dates)
        lead_seconds = np.asarray(self._file.variables["lead_time"][:], dtype=np.int64)
        valid_seconds = reference_seconds[:, None] + lead_seconds[None, :]

        self._file.variables["forecast_reference_time"][start:end] = reference_seconds
        self._file.variables["valid_time"][start:end, :] = valid_seconds
        for channel_idx, variable_name in enumerate(
            self._dataset.target.variable_names
        ):
            self._file.variables[variable_name][start:end, :, :, :] = prediction[
                :, :, channel_idx, :, :
            ].astype(np.float32, copy=False)

        self._sample_offset = end

    def _close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def on_test_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Close the NetCDF file after evaluation."""
        if not self.enabled:
            return
        self._close()
        logger.info(
            "Saved %d model prediction window(s) to %s.",
            self._sample_offset,
            self.output_path,
        )

    def teardown(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        stage: str,  # noqa: ARG002
    ) -> None:
        """Close an open output file if evaluation exits early."""
        self._close()
