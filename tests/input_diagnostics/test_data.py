"""Tests for icenet_mp.input_diagnostics.data."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from icenet_mp.input_diagnostics.data import build_sample_matrix, load_land_mask


class FakeSingleDataset:
    """Minimal in-memory dataset for sample-matrix tests."""

    def __init__(
        self,
        name: str,
        variables: list[str],
        values: dict[np.datetime64, np.ndarray],
    ) -> None:
        """Initialise a dataset from a date-to-array mapping."""
        self.name = name
        self.variable_names = variables
        self._values = values
        self.dates = sorted(values)
        self.frequency = np.timedelta64(1, "D")

    def get_tchw(self, dates: list[np.datetime64]) -> np.ndarray:
        return np.stack([self._values[date] for date in dates])


class TestBuildSampleMatrix:
    """Tests for the build_sample_matrix function."""

    def test_land_mask_excludes_masked_cells(self) -> None:
        """Passing land_mask must drop the land cell from the spatial mean."""
        d0 = np.datetime64("2020-01-01")
        d1 = np.datetime64("2020-01-02")
        # Land cell (bottom-right) is a huge outlier so masked vs unmasked means differ.
        values = {
            d0: np.array([[[10.0, 10.0], [10.0, 1000.0]]]),
            d1: np.array([[[20.0, 20.0], [20.0, 2000.0]]]),
        }
        dataset = FakeSingleDataset("sic", ["ice_conc"], values)
        land_mask = np.array([[True, True], [True, False]])

        matrix, names = build_sample_matrix({"sic": dataset}, land_mask=land_mask)  # type: ignore[dict-item]

        assert names == ["sic/ice_conc"]
        np.testing.assert_array_equal(matrix[:, 0], [10.0, 20.0])

    def test_no_land_mask_includes_all_cells(self) -> None:
        """Without land_mask, the spatial mean covers the full grid (existing behaviour)."""
        d0 = np.datetime64("2020-01-01")
        values = {d0: np.array([[[10.0, 10.0], [10.0, 1000.0]]])}
        dataset = FakeSingleDataset("sic", ["ice_conc"], values)

        matrix, _ = build_sample_matrix({"sic": dataset})  # type: ignore[dict-item]

        assert matrix[0, 0] == pytest.approx(257.5)


class TestLoadLandMask:
    """Tests for the load_land_mask function."""

    def test_returns_none_when_unset(self, caplog: pytest.LogCaptureFixture) -> None:
        """No rf.spatial.mask_dataset_name configured means no masking, with a warning."""
        config = OmegaConf.create({"base_path": "/tmp", "rf": {}})  # noqa: S108

        with caplog.at_level("WARNING"):
            result = load_land_mask(config)

        assert result is None
        assert "mask_dataset_name" in caplog.text

    def test_raises_when_configured_but_missing(self, tmp_path: object) -> None:
        """A configured but missing mask must fail loudly, not fall back silently."""
        config = OmegaConf.create(
            {
                "base_path": str(tmp_path),
                "rf": {"spatial": {"mask_dataset_name": "does-not-exist"}},
            }
        )

        with pytest.raises(FileNotFoundError, match="does-not-exist"):
            load_land_mask(config)

    def test_loads_array_when_present(self, tmp_path: object) -> None:
        """A configured, existing mask should load as a boolean array."""
        from pathlib import Path  # noqa: PLC0415

        base_path = Path(str(tmp_path))
        mask_path = base_path / "data" / "preprocessing" / "masks" / "some-dataset"
        mask_path.mkdir(parents=True)
        expected = np.array([[True, False], [True, True]])
        np.save(mask_path / "land_mask.npy", expected)

        config = OmegaConf.create(
            {
                "base_path": str(base_path),
                "rf": {"spatial": {"mask_dataset_name": "some-dataset"}},
            }
        )

        result = load_land_mask(config)

        assert result is not None
        np.testing.assert_array_equal(result, expected)
