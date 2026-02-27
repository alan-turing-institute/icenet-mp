import logging
import shutil
from pathlib import Path

import typer
from anemoi.datasets.commands.finalise import Finalise
from anemoi.datasets.commands.init import Init
from anemoi.datasets.commands.inspect import InspectZarr
from anemoi.datasets.commands.load import Load
from anemoi.datasets.data.dataset import Dataset as AnemoiDataset
from omegaconf import DictConfig, OmegaConf
from zarr.core import Array as ZarrArray
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
            download_in_progress, download_complete, statistics_ready = self.status()
            # The dataset is being downloaded
            if download_in_progress:
                logger.warning(
                    "Dataset %s at %s is currently being downloaded by another process.",
                    self.name,
                    self.path_dataset,
                )
                return
            # If the download is complete then check whether the dataset is valid
            if download_complete:
                # If the statistics are not ready we should finalise
                if not statistics_ready:
                    self.finalise()

                # Inspect the dataset for validity
                try:
                    self.inspect()
                    logger.info(
                        "Dataset %s at %s has been downloaded and seems to be valid.",
                        self.name,
                        self.path_dataset,
                    )
                except (AttributeError, FileNotFoundError, PathNotFoundError):
                    # If the dataset is invalid we flag this to the user and exit
                    logger.error(  # noqa: TRY400
                        "Dataset %s at %s seems to be invalid. Please check manually.",
                        self.name,
                        self.path_dataset,
                    )
                    typer.Exit(1)
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
        if download_complete and (not download_in_progress) and (not statistics_ready):
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
        logger.info("Finalised dataset %s at %s.", self.name, self.path_dataset)

    def initialise(self) -> None:
        """Initialise an Anemoi dataset."""
        if self.path_dataset.exists():
            logger.info(
                "Dataset %s at %s is already initialised.", self.name, self.path_dataset
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
        size: bool = True,
        statistics: bool = False,
    ) -> None:
        """Inspect an Anemoi dataset."""
        logger.info("Inspecting dataset %s at %s.", self.name, self.path_dataset)
        if self.path_dataset.exists():
            InspectZarr().run(
                AnemoiInspectArgs(
                    path=str(self.path_dataset),
                    detailed=detailed,
                    progress=(not detailed),
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
        try:
            ds_info = InspectZarr()._info(str(self.path_dataset))
            download_in_progress = ds_info.copy_in_progress
            if isinstance(dataset := ds_info.dataset, AnemoiDataset) and isinstance(
                array := ds_info.data, ZarrArray
            ):
                n_dates_expected = len(dataset.dates) - len(dataset.missing)
                n_dates_in_zarr = array.nchunks_initialized
                download_complete = n_dates_expected == n_dates_in_zarr
            else:
                download_complete = False
            statistics_ready = ds_info.statistics_ready
        except (AttributeError, FileNotFoundError, PathNotFoundError):
            download_in_progress = False
            download_complete = False
            statistics_ready = False
        return (download_in_progress, download_complete, statistics_ready)
