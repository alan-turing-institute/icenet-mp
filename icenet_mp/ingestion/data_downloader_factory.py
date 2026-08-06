from types import MappingProxyType

from omegaconf import DictConfig

from .data_downloader import DataDownloader
from .preprocessors import IceNetSICPreprocessor, NullPreprocessor


class DataDownloaderFactory:
    preprocessors = MappingProxyType(
        {
            "None": NullPreprocessor,
            "IceNetSIC": IceNetSICPreprocessor,
        }
    )

    def __init__(self, config: DictConfig) -> None:
        """Initialise a DataDownloaderFactory from a config."""
        self.downloaders: list[DataDownloader] = []
        for dataset_name in config["data"]["datasets"]:
            cls_preprocessor = self.preprocessors[
                config["data"]["datasets"][dataset_name]
                .get("preprocessor", {})
                .get("type", "None")
            ]
            self.downloaders.append(
                DataDownloader(dataset_name, config, cls_preprocessor)  # type: ignore[type-abstract]
            )
