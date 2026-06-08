from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import anemoi.datasets.create
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
def mock_creator(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch anemoi creator_factory to return a MagicMock for testing."""
    creator = MagicMock()

    def mock_factory(name: str, **kwargs: Any) -> MagicMock:
        """A mock creator factory that captures the task name, config, and path."""
        creator.task_name = name
        creator.config = kwargs.get("config")
        creator.path = kwargs.get("path")
        return creator

    monkeypatch.setattr(anemoi.datasets.create, "creator_factory", mock_factory)
    return creator


class TestDataDownloader:
    """Verify that DataDownloader delivers the right config and path to the anemoi creator."""

    def test_initialise_config_and_path_reach_creator(
        self,
        mock_downloader: DataDownloader,
        mock_creator: MagicMock,
    ) -> None:
        mock_downloader.initialise()
        assert mock_creator.task_name == "init"
        assert mock_creator.config is mock_downloader.config
        assert mock_creator.path == str(mock_downloader.path_dataset)

    def test_finalise_config_and_path_reach_creator(
        self,
        mock_downloader: DataDownloader,
        mock_creator: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(mock_downloader, "create_masks", MagicMock())
        mock_downloader.finalise(overwrite=False)
        assert mock_creator.task_name == "finalise"
        assert mock_creator.config is mock_downloader.config
        assert mock_creator.path == str(mock_downloader.path_dataset)

    def test_load_in_chunks_config_and_path_reach_creator(
        self,
        mock_downloader: DataDownloader,
        mock_creator: MagicMock,
    ) -> None:
        mock_downloader.load_in_chunks()
        assert mock_creator.task_name == "load"
        assert mock_creator.config is mock_downloader.config
        assert mock_creator.path == str(mock_downloader.path_dataset)
