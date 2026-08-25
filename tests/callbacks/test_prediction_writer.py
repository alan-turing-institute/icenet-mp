from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from netCDF4 import Dataset as NetCDFDataset

from icenet_mp.callbacks import PredictionWriter
from icenet_mp.data import CombinedDataset
from icenet_mp.types import DataSpace


def _combined_dataset() -> CombinedDataset:
    dataset = cast("Any", object.__new__(CombinedDataset))
    dataset.n_forecast_steps = 2
    dataset.n_history_steps = 2
    dataset.frequency = np.timedelta64(1, "D")
    dataset.dates = [
        np.datetime64("2026-01-01T12:00:00"),
        np.datetime64("2026-01-02T12:00:00"),
        np.datetime64("2026-01-03T12:00:00"),
    ]
    dataset.target = SimpleNamespace(
        hemisphere="north",
        latitudes=[80.0, 80.0, 79.0, 79.0],
        longitudes=[0.0, 1.0, 0.0, 1.0],
        space=DataSpace(channels=1, name="sic-osisaf", shape=(2, 2)),
        statistics={
            "minimum": np.asarray([0.2], dtype=np.float32),
            "maximum": np.asarray([0.8], dtype=np.float32),
        },
        variable_names=["ice_conc"],
    )
    return cast("CombinedDataset", dataset)


def _trainer(dataset: CombinedDataset, *, world_size: int = 1) -> Any:  # noqa: ANN401
    return SimpleNamespace(
        test_dataloaders=SimpleNamespace(dataset=dataset),
        world_size=world_size,
    )


class TestPredictionWriter:
    def test_disabled_writer_is_a_noop(self) -> None:
        writer = PredictionWriter()

        writer.on_test_start(SimpleNamespace(world_size=4), None)  # type: ignore[arg-type]
        writer.on_test_batch_end(  # type: ignore[arg-type]
            SimpleNamespace(),
            None,
            None,
            None,
            0,
        )
        writer.on_test_end(SimpleNamespace(), None)  # type: ignore[arg-type]

    def test_rejects_multi_process_export(self, tmp_path: Path) -> None:
        writer = PredictionWriter(tmp_path / "predictions.nc")

        with pytest.raises(RuntimeError, match="single-process evaluation"):
            writer.on_test_start(  # type: ignore[arg-type]
                _trainer(_combined_dataset(), world_size=2),
                None,
            )

    def test_writes_denormalised_predictions_and_coordinates(
        self, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "predictions.nc"
        dataset = _combined_dataset()
        trainer = _trainer(dataset)
        writer = PredictionWriter(output_path)

        writer.on_test_start(trainer, None)  # type: ignore[arg-type]

        first_batch = torch.full((2, 2, 1, 2, 2), 0.5, dtype=torch.float32)
        second_batch = torch.ones((1, 2, 1, 2, 2), dtype=torch.float32)

        writer.on_test_batch_end(  # type: ignore[arg-type]
            trainer,
            None,
            {"prediction": first_batch},
            None,
            0,
        )
        writer.on_test_batch_end(  # type: ignore[arg-type]
            trainer,
            None,
            {"prediction": second_batch},
            None,
            1,
        )
        writer.on_test_end(trainer, None)  # type: ignore[arg-type]

        with NetCDFDataset(output_path) as netcdf:
            assert netcdf.getncattr("Conventions") == "CF-1.10"
            assert netcdf.getncattr("hemisphere") == "north"
            assert len(netcdf.dimensions["forecast_reference_time"]) == 3
            assert len(netcdf.dimensions["lead_time"]) == 2

            prediction = np.asarray(netcdf.variables["ice_conc"][:])
            assert prediction.shape == (3, 2, 2, 2)
            assert np.allclose(prediction[:2], 0.5)
            assert np.allclose(prediction[2:], 0.8)
            assert (
                netcdf.variables["ice_conc"].standard_name
                == "sea_ice_area_fraction"
            )

            assert np.array_equal(
                np.asarray(netcdf.variables["lead_time"][:]),
                np.asarray([86400, 172800]),
            )

            reference = np.asarray(netcdf.variables["forecast_reference_time"][:])
            expected_reference = np.asarray(
                [
                    np.datetime64("2026-01-02T12:00:00"),
                    np.datetime64("2026-01-03T12:00:00"),
                    np.datetime64("2026-01-04T12:00:00"),
                ],
            ).astype("datetime64[s]").astype(np.int64)
            assert np.array_equal(reference, expected_reference)

            valid = np.asarray(netcdf.variables["valid_time"][:])
            assert np.array_equal(
                valid[0],
                expected_reference[0] + np.asarray([86400, 172800]),
            )
            assert np.allclose(
                np.asarray(netcdf.variables["latitude"][:]),
                np.asarray([[80.0, 80.0], [79.0, 79.0]]),
            )
            assert np.allclose(
                np.asarray(netcdf.variables["longitude"][:]),
                np.asarray([[0.0, 1.0], [0.0, 1.0]]),
            )

    def test_rejects_prediction_channel_mismatch(self, tmp_path: Path) -> None:
        dataset = _combined_dataset()
        trainer = _trainer(dataset)
        writer = PredictionWriter(tmp_path / "predictions.nc")
        writer.on_test_start(trainer, None)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="channel count"):
            writer.on_test_batch_end(  # type: ignore[arg-type]
                trainer,
                None,
                {"prediction": torch.zeros((1, 2, 2, 2, 2))},
                None,
                0,
            )

        writer.teardown(trainer, None, "test")  # type: ignore[arg-type]
