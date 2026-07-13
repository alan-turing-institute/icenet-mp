import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import anemoi.datasets.create.tasks
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from icenet_mp.data_processors.data_downloader import DataDownloader
from icenet_mp.types import AnemoiDatasetStatus, AnemoiInspectArgs
from tests.conftest import build_zarr


@dataclass
class MockDsInfo:
    """Configurable stand-in for the object returned by InspectZarr()._info()."""

    copy_in_progress: bool = False
    statistics_ready: bool = False
    dataset: object = field(default_factory=object)
    build_flags: list[bool] | None = None
    statistics_started: str | None = None


class MockInspectZarr:
    """Stand-in for InspectZarr; _info() and run() are pre-configured or raise."""

    def __init__(self) -> None:
        """Initialise a MockInspectZarr."""
        self.ds_info: MockDsInfo | BaseException = MockDsInfo()
        self.run_error: BaseException | None = None
        self.run_args: AnemoiInspectArgs | None = None

    def _info(self, _: str) -> MockDsInfo:
        if isinstance(self.ds_info, BaseException):
            raise self.ds_info
        return self.ds_info

    def run(self, args: AnemoiInspectArgs) -> None:
        self.run_args = args
        if self.run_error is not None:
            raise self.run_error


def _make_data_downloader(tmp_path: Path, name: str) -> DataDownloader:
    """Build a DataDownloader for `name` from a minimal dataset config."""
    full_cfg: DictConfig = OmegaConf.create(
        {
            "base_path": str(tmp_path),
            "data": {
                "datasets": {
                    name: {
                        "name": name,
                        "preprocessor": {"type": "dummy"},
                        "dates": {
                            "start": "2020-01-01",
                            "end": "2020-01-31",
                            "frequency": "24h",
                        },
                    }
                },
            },
        }
    )
    return DataDownloader(name, full_cfg, MagicMock)


@pytest.fixture
def mock_data_downloader(tmp_path: Path) -> DataDownloader:
    """A DataDownloader built from a minimal dataset config."""
    return _make_data_downloader(tmp_path, "test")


@pytest.fixture
def mock_data_downloader_ssmis(tmp_path: Path) -> DataDownloader:
    """A DataDownloader for an SSMIS dataset, used to test mask generation."""
    return _make_data_downloader(tmp_path, "sic-ssmis")


@pytest.fixture
def mock_dispatcher(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch anemoi TaskDispatcher to return a MagicMock for testing."""
    dispatcher = MagicMock()

    def mock_init(creator: object) -> MagicMock:
        dispatcher.creator = creator
        return dispatcher

    monkeypatch.setattr(anemoi.datasets.create.tasks, "TaskDispatcher", mock_init)
    return dispatcher


@pytest.fixture
def mock_inspect_zarr(monkeypatch: pytest.MonkeyPatch) -> MockInspectZarr:
    """Patch InspectZarr with MockInspectZarr; returns the instance for per-test configuration."""
    instance = MockInspectZarr()
    monkeypatch.setattr(
        "icenet_mp.data_processors.data_downloader.InspectZarr",
        lambda: instance,
    )
    return instance


class TestDataDownloader:
    """Verify that DataDownloader delivers the right recipe and path to the anemoi creator."""

    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2020, 1, 31)
    frequency = datetime.timedelta(hours=24)

    def test_initialise_recipe_and_path_reach_creator(
        self,
        mock_data_downloader: DataDownloader,
        mock_dispatcher: MagicMock,
    ) -> None:
        mock_data_downloader.initialise()
        mock_dispatcher.task_init.assert_called_once()
        assert mock_dispatcher.creator.recipe.dates.start == self.start_date
        assert mock_dispatcher.creator.recipe.dates.end == self.end_date
        assert mock_dispatcher.creator.recipe.dates.frequency == self.frequency
        assert mock_dispatcher.creator.path == str(mock_data_downloader.path_dataset)

    def test_finalise_recipe_and_path_reach_creator(
        self,
        mock_data_downloader: DataDownloader,
        mock_dispatcher: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(mock_data_downloader, "generate_masks", MagicMock())
        status = AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=False
        )
        mock_data_downloader.finalise(overwrite=False, status=status)
        mock_dispatcher.task_finalise.assert_called_once()
        assert mock_dispatcher.creator.recipe.dates.start == self.start_date
        assert mock_dispatcher.creator.recipe.dates.end == self.end_date
        assert mock_dispatcher.creator.recipe.dates.frequency == self.frequency
        assert mock_dispatcher.creator.path == str(mock_data_downloader.path_dataset)

    def test_load_in_chunks_recipe_and_path_reach_creator(
        self,
        mock_data_downloader: DataDownloader,
        mock_dispatcher: MagicMock,
    ) -> None:
        mock_data_downloader.load_in_chunks()
        mock_dispatcher.task_load.assert_called_once()
        assert mock_dispatcher.creator.recipe.dates.start == self.start_date
        assert mock_dispatcher.creator.recipe.dates.end == self.end_date
        assert mock_dispatcher.creator.recipe.dates.frequency == self.frequency
        assert mock_dispatcher.creator.path == str(mock_data_downloader.path_dataset)

    def test_copy_in_progress_not_complete(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(copy_in_progress=True)
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=True, download_complete=False, is_finalised=False
        )

    def test_dataset_none_not_complete(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(dataset=None)
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=False, is_finalised=False
        )

    def test_build_flags_all_true_statistics_ready(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(
            statistics_ready=True, build_flags=[True, True, True]
        )
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=True
        )

    def test_build_flags_all_true_statistics_not_ready(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(build_flags=[True, True, True])
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=False
        )

    def test_build_flags_empty_not_complete(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(build_flags=[])
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=False, is_finalised=False
        )

    def test_build_flags_partial_not_complete(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(build_flags=[True, False, True])
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=False, is_finalised=False
        )

    def test_no_build_flags_statistics_ready(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(statistics_ready=True)
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=True
        )

    def test_no_build_flags_statistics_started_not_finalised(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(statistics_started="2020-01-31")
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=False
        )

    def test_no_build_flags_no_statistics_raises(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(statistics_started=None)
        with pytest.raises(RuntimeError, match="Unable to determine readiness"):
            mock_data_downloader.check_status()

    def test_file_not_found_raises(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = FileNotFoundError("no dataset at path")
        with pytest.raises(RuntimeError, match="Unable to get status"):
            mock_data_downloader.check_status()

    def test_attribute_error_raises(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = AttributeError("unexpected ds_info shape")
        with pytest.raises(RuntimeError, match="Unable to get status"):
            mock_data_downloader.check_status()

    def test_inspect_verbose_calls_inspect_zarr_run(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        """inspect(verbose=True) reaches the real InspectZarr().run() call."""
        mock_data_downloader.path_dataset.mkdir(parents=True)
        mock_data_downloader.inspect(verbose=True)
        assert mock_inspect_zarr.run_args is not None
        args = mock_inspect_zarr.run_args
        assert args.path == str(mock_data_downloader.path_dataset)
        assert args.detailed is True
        assert args.progress is False
        assert args.statistics is False
        assert args.size is True

    def test_inspect_verbose_suppresses_value_error(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """inspect(verbose=True) does not raise when InspectZarr().run() raises ValueError."""
        mock_data_downloader.path_dataset.mkdir(parents=True)
        mock_inspect_zarr.run_error = ValueError("statistics not available")
        with caplog.at_level(logging.WARNING):
            mock_data_downloader.inspect(verbose=True)
        assert "Further dataset information unavailable" in caplog.text

    def test_inspect_verbose_wraps_file_not_found_error(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        """inspect(verbose=True) wraps FileNotFoundError from InspectZarr().run() as RuntimeError."""
        mock_data_downloader.path_dataset.mkdir(parents=True)
        mock_inspect_zarr.run_error = FileNotFoundError("no dataset at path")
        with pytest.raises(RuntimeError, match="Failed to load dataset"):
            mock_data_downloader.inspect(verbose=True)

    def test_finalise_suppresses_cleanup_value_error_and_logs_residual(
        self,
        mock_data_downloader: DataDownloader,
        mock_data: dict[str, dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test the ValueError() suppression in finalise() by providing a dataset with no recipe."""
        monkeypatch.setattr(mock_data_downloader, "generate_masks", MagicMock())
        build_zarr(mock_data_downloader.path_dataset, mock_data)
        residual = mock_data_downloader.path_dataset.parent / (
            f"{mock_data_downloader.path_dataset.stem}.tmp"
        )
        residual.mkdir()
        (residual / "dummy.txt").write_text("x")

        status = AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=True
        )
        with caplog.at_level(logging.WARNING):
            mock_data_downloader.finalise(overwrite=False, status=status)

        # Cleanup's real ValueError is suppressed, so the residual artifact remains
        # and is logged as a warning rather than raising.
        # The artifa
        assert residual.exists()
        assert "Residual artifacts" in caplog.text

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_generate_masks_creates_or_recreates_land_and_active_masks(
        self,
        mock_data_downloader_ssmis: DataDownloader,
        mock_data_status_flag: dict[str, dict[str, Any]],
        *,
        overwrite: bool,
    ) -> None:
        build_zarr(mock_data_downloader_ssmis.path_dataset, mock_data_status_flag)
        land_mask_path = mock_data_downloader_ssmis.path_masks / "land_mask.npy"
        active_mask_path = mock_data_downloader_ssmis.path_masks / "active_mask.npy"
        if overwrite:
            # Pre-seed masks with sentinel values to prove overwrite=True recomputes them.
            mock_data_downloader_ssmis.path_masks.mkdir(parents=True)
            np.save(land_mask_path, np.full((2, 2), fill_value=9))
            np.save(active_mask_path, np.full((2, 2), fill_value=9))

        mock_data_downloader_ssmis.generate_masks(overwrite=overwrite)

        np.testing.assert_array_equal(np.load(land_mask_path), [[0, 1], [1, 1]])
        np.testing.assert_array_equal(np.load(active_mask_path), [[0, 1], [0, 1]])

    def test_generate_masks_skips_non_ssmis_dataset(
        self, mock_data_downloader: DataDownloader
    ) -> None:
        """No dataset exists on disk; a real open_dataset() call would raise if reached."""
        mock_data_downloader.generate_masks(overwrite=False)
        assert not mock_data_downloader.path_masks.exists()

    def test_generate_masks_skips_when_masks_already_exist(
        self, mock_data_downloader_ssmis: DataDownloader
    ) -> None:
        mock_data_downloader_ssmis.path_masks.mkdir(parents=True)
        land_mask_path = mock_data_downloader_ssmis.path_masks / "land_mask.npy"
        active_mask_path = mock_data_downloader_ssmis.path_masks / "active_mask.npy"
        np.save(land_mask_path, np.zeros((2, 2)))
        np.save(active_mask_path, np.ones((2, 2)))

        # No dataset exists on disk; a real open_dataset() call would raise if reached.
        mock_data_downloader_ssmis.generate_masks(overwrite=False)

        np.testing.assert_array_equal(np.load(land_mask_path), np.zeros((2, 2)))
        np.testing.assert_array_equal(np.load(active_mask_path), np.ones((2, 2)))
