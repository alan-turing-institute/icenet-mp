from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

import icenet_mp.cli.datasets as datasets_module
from icenet_mp.ingestion.data_downloader import DataDownloader


class FakeDataset:
    name = "example"
    hemisphere = "north"
    dates = [np.datetime64("2020-01-01")]
    variable_names = ["ice_conc", "temperature"]

    def __init__(self, **_kwargs) -> None:  # noqa: ANN003
        pass

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _idx: int) -> np.ndarray:
        return np.ones((2, 4, 4), dtype=np.float32)


def test_plot_downloader_saves_each_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "example.zarr"
    dataset_path.mkdir()

    class FakeDownloader:
        name = "example"
        path_dataset = dataset_path

    monkeypatch.setattr(datasets_module, "SingleDataset", FakeDataset)

    def fake_plot_static_inputs(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        return {
            "2020-01-01-example:ice_conc": [Image.new("RGB", (4, 4))],
            "2020-01-01-example:temperature": [Image.new("RGB", (4, 4))],
        }

    monkeypatch.setattr(datasets_module, "plot_static_inputs", fake_plot_static_inputs)

    output_dir = tmp_path / "plots"
    saved = datasets_module._plot_downloader(
        cast("DataDownloader", FakeDownloader()), output_dir, 0
    )

    assert saved == 2
    assert (output_dir / "example" / "2020-01-01-example_ice_conc.png").is_file()
    assert (output_dir / "example" / "2020-01-01-example_temperature.png").is_file()


def test_plot_downloader_rejects_out_of_range_timestep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "example.zarr"
    dataset_path.mkdir()

    class FakeDownloader:
        name = "example"
        path_dataset = dataset_path

    monkeypatch.setattr(datasets_module, "SingleDataset", FakeDataset)

    with pytest.raises(IndexError, match="Timestep 2 is out of range"):
        datasets_module._plot_downloader(
            cast("DataDownloader", FakeDownloader()), tmp_path / "plots", 2
        )
