import io
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from icenet_mp.visualisations import plot_variables_static, plot_variables_video


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


class FakeVideoDataset:
    """Minimal dataset stub for video plotting tests."""

    name = "example"
    hemisphere = "north"
    dates: ClassVar[list[np.datetime64]] = [
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-02"),
        np.datetime64("2020-01-03"),
    ]
    variable_names: ClassVar[list[str]] = ["ice_conc", "temperature"]

    def __init__(self, **_kwargs) -> None:  # noqa: ANN003
        """Accept the constructor arguments used by the plotting helper."""

    def __len__(self) -> int:
        """Return the three available timesteps."""
        return 3

    def get_tchw_slice(self, _start_date: np.datetime64, n_steps: int) -> np.ndarray:
        """Return deterministic values for the requested number of timesteps."""
        return np.ones((n_steps, 2, 4, 4), dtype=np.float32)


class TestPlotDataset:
    def test_plot_dataset_saves_each_variable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save one PNG for each variable returned by the plotting helper."""
        dataset_path = tmp_path / "example.zarr"
        dataset_path.mkdir()

        monkeypatch.setattr(
            "icenet_mp.visualisations.dataset_plotting.SingleDataset", FakeDataset
        )

        def fake_plot_static_inputs(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return {
                "2020-01-01-example:ice_conc": [Image.new("RGB", (4, 4))],
                "2020-01-01-example:temperature": [Image.new("RGB", (4, 4))],
            }

        monkeypatch.setattr(
            "icenet_mp.visualisations.dataset_plotting.plot_static_inputs",
            fake_plot_static_inputs,
        )

        output_dir = tmp_path / "plots"
        saved = plot_variables_static("example", dataset_path, output_dir, 0)

        assert saved == 2
        assert (output_dir / "example" / "2020-01-01-example_ice_conc.png").is_file()
        assert (output_dir / "example" / "2020-01-01-example_temperature.png").is_file()

    def test_plot_dataset_rejects_out_of_range_timestep(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject a timestep index outside the available dataset range."""
        dataset_path = tmp_path / "example.zarr"
        dataset_path.mkdir()

        monkeypatch.setattr(
            "icenet_mp.visualisations.dataset_plotting.SingleDataset", FakeDataset
        )

        with pytest.raises(IndexError, match="Timestep 2 is out of range"):
            plot_variables_static("example", dataset_path, tmp_path / "plots", 2)


class TestPlotDatasetVideo:
    def test_plot_dataset_video_saves_each_variable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save one video for each variable returned by the plotting helper."""
        dataset_path = tmp_path / "example.zarr"
        dataset_path.mkdir()

        monkeypatch.setattr(
            "icenet_mp.visualisations.dataset_plotting.SingleDataset", FakeVideoDataset
        )

        def fake_plot_video_inputs(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return {
                "2020-01-01-example:ice_conc": io.BytesIO(b"ice_conc video"),
                "2020-01-01-example:temperature": io.BytesIO(b"temperature video"),
            }

        monkeypatch.setattr(
            "icenet_mp.visualisations.dataset_plotting.plot_video_inputs",
            fake_plot_video_inputs,
        )

        output_dir = tmp_path / "plots"
        saved = plot_variables_video("example", dataset_path, output_dir, 0, 3)

        assert saved == 2
        assert (output_dir / "example" / "2020-01-01-example_ice_conc.mp4").is_file()
        assert (output_dir / "example" / "2020-01-01-example_temperature.mp4").is_file()

    def test_plot_dataset_video_rejects_out_of_range_timestep(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject a timestep/n_steps combination outside the dataset range."""
        dataset_path = tmp_path / "example.zarr"
        dataset_path.mkdir()

        monkeypatch.setattr(
            "icenet_mp.visualisations.dataset_plotting.SingleDataset", FakeVideoDataset
        )

        with pytest.raises(IndexError, match="Timesteps 2:5 are out of range"):
            plot_variables_video("example", dataset_path, tmp_path / "plots", 2, 3)
