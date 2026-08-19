from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import pytest
from PIL import Image

import icenet_mp.cli.datasets as datasets_module

if TYPE_CHECKING:
    from icenet_mp.ingestion.data_downloader import DataDownloader


class FakeDataset:
    """Minimal dataset stub for plotting tests."""

    name = "example"
    hemisphere = "north"
    dates: ClassVar[list[np.datetime64]] = [np.datetime64("2020-01-01")]
    variable_names: ClassVar[list[str]] = ["ice_conc", "temperature"]

    def __init__(self, **_kwargs) -> None:  # noqa: ANN003
        """Accept the constructor arguments used by the plotting helper."""

    def __len__(self) -> int:
        """Return the single available timestep."""
        return 1

    def __getitem__(self, _idx: int) -> np.ndarray:
        """Return two deterministic variables for the requested timestep."""
        return np.ones((2, 4, 4), dtype=np.float32)


def test_plot_downloader_saves_each_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save one PNG for each variable returned by the plotting helper."""
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
    """Reject a timestep index outside the available dataset range."""
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
