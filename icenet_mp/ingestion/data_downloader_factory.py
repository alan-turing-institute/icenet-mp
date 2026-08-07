from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from .data_downloader import DataDownloader
from .postprocessors import CompositePostprocessor
from .preprocessors import CompositePreprocessor


class DataDownloaderFactory:
    def __init__(self, config: DictConfig) -> None:
        """Initialise a DataDownloaderFactory from a config."""
        self.downloaders: list[DataDownloader] = []
        base_path = Path(config["base_path"]).resolve()
        for dataset_name, dataset_config in config["data"]["datasets"].items():
            # Anemoi 'forcings' need to be escaped with `\${}` to avoid being resolved here
            anemoi_config = cast("dict[str, Any]", OmegaConf.to_object(dataset_config))
            self.downloaders.append(
                DataDownloader(
                    dataset_name,
                    base_path,
                    anemoi_config,
                    CompositePreprocessor(dataset_name, dataset_config, base_path),
                    CompositePostprocessor(dataset_name, dataset_config, base_path),
                )
            )
