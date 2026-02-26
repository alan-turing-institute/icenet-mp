import logging
import shutil
from pathlib import Path

from anemoi.datasets.commands.finalise import Finalise
from anemoi.datasets.commands.init import Init
from anemoi.datasets.commands.inspect import InspectZarr
from anemoi.datasets.commands.load import Load
from omegaconf import DictConfig, OmegaConf
from zarr.errors import PathNotFoundError

from icenet_mp.types import (
    AnemoiFinaliseArgs,
    AnemoiInitArgs,
    AnemoiInspectArgs,
    AnemoiLoadArgs,
)

from .preprocessors import IPreprocessor

logger = logging.getLogger(__name__)


class DataDownloader:
    def __init__(
        self, name: str, config: DictConfig, cls_preprocessor: type[IPreprocessor]
    ) -> None:
        """Initialise a DataDownloader from a config.

        Register a preprocessor if appropriate.
        """
        self.name = name
        _data_path = Path(config["base_path"]).resolve() / "data"
        self.path_dataset = _data_path / "anemoi" / f"{name}.zarr"
        self.path_preprocessor = _data_path / "preprocessing"
        # Note that Anemoi 'forcings' need to be escaped with `\${}` to avoid being resolved here
        self.config: DictConfig = OmegaConf.to_object(config["data"]["datasets"][name])  # type: ignore[assignment]
        self.preprocessor = cls_preprocessor(self.config)

    def create(self, *, overwrite: bool) -> None:
        """Ensure that a single Anemoi dataset exists."""
        # If we are overwriting we delete any existing dataset
        if overwrite:
            logger.info(
                "Overwrite set to true, redownloading %s to %s",
                self.name,
                self.path_dataset,
            )
            shutil.rmtree(self.path_dataset, ignore_errors=True)

        # Otherwise we check whether a valid dataset exists
        elif self.path_dataset.exists():
            download_in_progress, _, _ = self.status()
            # The dataset is being downloaded
            if download_in_progress:
                logger.warning(
                    "Dataset %s at %s is currently being downloaded by another process. Please wait until it is complete.",
                    self.name,
                    self.path_dataset,
                )
                return
            # Check whether the dataset is valid, even if it is incomplete
            try:
                self.inspect()
                logger.info(
                    "Dataset %s already exists at %s, no need to download.",
                    self.name,
                    self.path_dataset,
                )
            except (AttributeError, FileNotFoundError, PathNotFoundError):
                # If the dataset is invalid we delete it
                logger.info(
                    "Dataset %s at %s is invalid, removing it.",
                    self.name,
                    self.path_dataset,
                )
                shutil.rmtree(self.path_dataset, ignore_errors=True)
            else:
                # If the dataset is valid we return here
                return

        # Download the dataset
        self.download()

    def download(self) -> None:
        """Download an Anemoi dataset in parts."""
        self.preprocessor.download(self.path_preprocessor)
        logger.info("Creating dataset %s at %s.", self.name, self.path_dataset)
        # Initialise
        self.initialise()
        # Load in parts
        self.load_in_chunks()
        # Finalise if the status indicates the dataset is complete
        download_in_progress, download_complete, statistics_ready = self.status()
        if download_complete and (not download_in_progress) and statistics_ready:
            self.finalise()
        else:
            logger.warning(
                "Dataset %s at %s is not fully loaded, skipping finalise.",
                self.name,
                self.path_dataset,
            )

    def finalise(self) -> None:
        """Finalise the segmented Anemoi dataset."""
        Finalise().run(
            AnemoiFinaliseArgs(
                path=str(self.path_dataset),
                config=self.config,
            )
        )

    def initialise(self) -> None:
        """Initialise an Anemoi dataset."""
        if self.path_dataset.exists():
            logger.info(
                "Dataset %s already initialised at %s.", self.name, self.path_dataset
            )
            return
        try:
            Init().run(
                AnemoiInitArgs(
                    path=str(self.path_dataset),
                    config=self.config,
                )
            )
            logger.info("Initialised dataset %s at %s.", self.name, self.path_dataset)
        except (AttributeError, FileNotFoundError, PathNotFoundError):
            logger.exception(
                "Failed to initialise dataset %s at %s.",
                self.name,
                self.path_dataset,
            )
            raise

    def inspect(
        self,
        *,
        detailed: bool = True,
        progress: bool = True,
        statistics: bool = True,
        size: bool = True,
    ) -> None:
        """Inspect an Anemoi dataset."""
        logger.info("Inspecting dataset %s at %s.", self.name, self.path_dataset)
        if self.path_dataset.exists():
            InspectZarr().run(
                AnemoiInspectArgs(
                    path=str(self.path_dataset),
                    detailed=detailed,
                    progress=progress,
                    statistics=statistics,
                    size=size,
                )
            )
        else:
            logger.error("Dataset %s not found at %s.", self.name, self.path_dataset)

    def load_in_chunks(self) -> None:
        """Download a single Anemoi dataset in chunks, skipping those already present."""
        Load().run(
            AnemoiLoadArgs(
                path=str(self.path_dataset),
                config=self.config,
            )
        )

    def status(self) -> tuple[bool, bool, bool]:
        """Return a tuple indicating whether the dataset exists and whether it is complete."""
        inspector = InspectZarr()
        try:
            version = inspector._info(str(self.path_dataset))
            download_in_progress = version.copy_in_progress
            download_complete = all(version.build_flags or [])
            statistics_ready = version.statistics_ready
        except (AttributeError, FileNotFoundError, PathNotFoundError):
            download_in_progress = False
            download_complete = False
            statistics_ready = False
        return (download_in_progress, download_complete, statistics_ready)
