from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from icenet_mp.data_processors.data_downloader import DataDownloader
from icenet_mp.data_processors.preprocessors.ipreprocessor import IPreprocessor


class DummyPreprocessor(IPreprocessor):
    """A dummy preprocessor for testing."""

    def download(self, preprocessor_path: Path) -> None:  # pragma: no cover - unused
        preprocessor_path.mkdir(parents=True, exist_ok=True)


def _build_downloader(tmp_path: Path, dataset_cfg: dict) -> DataDownloader:
    """Helper to create a DataDownloader with a dummy preprocessor."""
    full_cfg: DictConfig = OmegaConf.create(
        {
            "base_path": str(tmp_path),
            "data": {
                "datasets": {
                    "test": {
                        "name": "test",
                        "preprocessor": {"type": "dummy"},
                        **dataset_cfg,
                    }
                },
            },
        }
    )
    return DataDownloader("test", full_cfg, DummyPreprocessor)


@pytest.fixture
def downloader_with_file_dataset(tmp_path: Path) -> DataDownloader:
    """Fixture that creates a downloader with a file-based dataset path."""
    downloader = _build_downloader(
        tmp_path,
        {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "group_by": "monthly",
        },
    )
    downloader.path_dataset = tmp_path / "test.zarr"
    # Create the file to ensure it exists
    downloader.path_dataset.touch()
    return downloader


@pytest.fixture
def downloader_with_directory_dataset(tmp_path: Path) -> DataDownloader:
    """Fixture that creates a downloader with a directory-based dataset path."""
    downloader = _build_downloader(
        tmp_path,
        {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "group_by": "monthly",
        },
    )
    downloader.path_dataset = tmp_path / "test_dir.zarr"
    downloader.path_dataset.mkdir(parents=True, exist_ok=True)
    return downloader
