"""Tests for writing synthetic frames as anemoi-compatible zarr stores."""

import datetime
from pathlib import Path

import numpy as np
import zarr

from icenet_mp.synthetic.zarr_writer import daily_dates, write_synthetic_zarr


class TestDailyDates:
    """`daily_dates` returns consecutive daily datetimes."""

    def test_length_and_spacing(self) -> None:
        """It returns `n_timesteps` dates exactly one day apart."""
        start = datetime.datetime(2021, 3, 1)
        dates = daily_dates(4, start)
        assert len(dates) == 4
        assert dates[0] == start
        assert dates[-1] == start + datetime.timedelta(days=3)


class TestWriteSyntheticZarr:
    """The written store matches the minimal anemoi on-disk layout."""

    def _frames(self) -> np.ndarray:
        return np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)

    def test_layout_and_attrs(self, tmp_path: Path) -> None:
        """Data is [T, 1, 1, H*W] and the expected attrs are recorded."""
        frames = self._frames()
        zarr_path = write_synthetic_zarr(
            tmp_path / "synth.zarr", frames=frames, variable_name="ice_conc"
        )
        store = zarr.open_group(str(zarr_path), mode="r")
        assert store["data"].shape == (3, 1, 1, 16)
        assert list(store.attrs["field_shape"]) == [4, 4]
        assert store.attrs["variables"] == ["ice_conc"]
        assert store.attrs["frequency"] == "24h"

    def test_dates_length_matches_frames(self, tmp_path: Path) -> None:
        """One date is written per frame."""
        frames = self._frames()
        zarr_path = write_synthetic_zarr(tmp_path / "synth.zarr", frames=frames)
        store = zarr.open_group(str(zarr_path), mode="r")
        assert store["dates"].shape == (3,)

    def test_stats_exclude_missing_dates(self, tmp_path: Path) -> None:
        """Statistics ignore frames on missing dates, even wildly out-of-range ones."""
        frames = np.ones((3, 4, 4), dtype=np.float32)
        frames[1] = 1000.0  # an outlier that must not skew the stats
        missing = [datetime.datetime(2020, 1, 2)]
        zarr_path = write_synthetic_zarr(
            tmp_path / "synth.zarr", frames=frames, missing_dates=missing
        )
        store = zarr.open_group(str(zarr_path), mode="r")
        assert float(store["maximum"][0]) == 1.0
        assert missing[0].isoformat(timespec="seconds") in store.attrs["missing_dates"]
