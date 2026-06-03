import datetime
from pathlib import Path
from unittest.mock import MagicMock

import anemoi.datasets.create.tasks
import pytest
from omegaconf import DictConfig, OmegaConf

from icenet_mp.data_processors.data_downloader import DataDownloader


@pytest.fixture
def mock_downloader(tmp_path: Path) -> DataDownloader:
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


class TestDataDownloader:
    """Verify that DataDownloader delivers the right recipe and path to the anemoi creator."""

    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2020, 1, 31)
    frequency = datetime.timedelta(hours=24)

    def test_initialise_recipe_and_path_reach_creator(
        self,
        mock_downloader: DataDownloader,
        mock_dispatcher: MagicMock,
    ) -> None:
        mock_downloader.initialise()
        mock_dispatcher.task_init.assert_called_once()
        assert mock_dispatcher.creator.recipe.dates.start == self.start_date
        assert mock_dispatcher.creator.recipe.dates.end == self.end_date
        assert mock_dispatcher.creator.recipe.dates.frequency == self.frequency
        assert mock_dispatcher.creator.path == str(mock_downloader.path_dataset)

    def test_finalise_recipe_and_path_reach_creator(
        self,
        mock_downloader: DataDownloader,
        mock_dispatcher: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(mock_downloader, "create_masks", MagicMock())
        mock_downloader.finalise(overwrite=False)
        mock_dispatcher.task_finalise.assert_called_once()
        assert mock_dispatcher.creator.recipe.dates.start == self.start_date
        assert mock_dispatcher.creator.recipe.dates.end == self.end_date
        assert mock_dispatcher.creator.recipe.dates.frequency == self.frequency
        assert mock_dispatcher.creator.path == str(mock_downloader.path_dataset)

    def test_load_in_chunks_recipe_and_path_reach_creator(
        self,
        mock_downloader: DataDownloader,
        mock_dispatcher: MagicMock,
    ) -> None:
        mock_downloader.load_in_chunks()
        mock_dispatcher.task_load.assert_called_once()
        assert mock_dispatcher.creator.recipe.dates.start == self.start_date
        assert mock_dispatcher.creator.recipe.dates.end == self.end_date
        assert mock_dispatcher.creator.recipe.dates.frequency == self.frequency
        assert mock_dispatcher.creator.path == str(mock_downloader.path_dataset)
