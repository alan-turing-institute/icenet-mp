import datetime
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import anemoi.datasets.create.tasks
import pytest
from omegaconf import DictConfig, OmegaConf

from icenet_mp.data_processors.data_downloader import DataDownloader
from icenet_mp.types import AnemoiDatasetStatus


@dataclass
class MockDsInfo:
    """Configurable stand-in for the object returned by InspectZarr()._info()."""

    copy_in_progress: bool = False
    statistics_ready: bool = False
    dataset: object = field(default_factory=object)
    build_flags: list[bool] | None = None
    statistics_started: str | None = None


class MockInspectZarr:
    """Stand-in for InspectZarr; _info() returns a pre-configured MockDsInfo or raises."""

    def __init__(self) -> None:
        """Initialise a MockInspectZarr."""
        self.ds_info: MockDsInfo | BaseException = MockDsInfo()

    def _info(self, _: str) -> MockDsInfo:
        if isinstance(self.ds_info, BaseException):
            raise self.ds_info
        return self.ds_info


@pytest.fixture
def mock_data_downloader(tmp_path: Path) -> DataDownloader:
    """A DataDownloader built from a minimal dataset config."""
    full_cfg: DictConfig = OmegaConf.create(
        {
            "base_path": str(tmp_path),
            "data": {
                "datasets": {
                    "test": {
                        "name": "test",
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
    return DataDownloader("test", full_cfg, MagicMock)


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
