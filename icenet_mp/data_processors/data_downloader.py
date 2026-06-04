import logging
import shutil
from pathlib import Path

import numpy as np
import typer
from anemoi.datasets.commands.finalise import Finalise
from anemoi.datasets.commands.init import Init
from anemoi.datasets.commands.inspect import InspectZarr
from anemoi.datasets.commands.load import Load
from anemoi.datasets.data import open_dataset
from omegaconf import DictConfig, OmegaConf
from zarr.errors import PathNotFoundError

from icenet_mp.types import (
    AnemoiDatasetStatus,
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
        self.path_masks = self.path_preprocessor / "masks" / name
        # Note that Anemoi 'forcings' need to be escaped with `\${}` to avoid being resolved here
        self.config: DictConfig = OmegaConf.to_object(config["data"]["datasets"][name])  # type: ignore[assignment]
        self.preprocessor = cls_preprocessor(self.config)

    def check_status(self) -> AnemoiDatasetStatus:
        """Return the status of the dataset."""
        try:
            ds_info = InspectZarr()._info(str(self.path_dataset))
            is_finalised = ds_info.statistics_ready
            if (copy_flags := ds_info.copy_flags) is not None:
                download_complete = bool(all(copy_flags))
            elif (build_flags := ds_info.build_flags) is not None:
                download_complete = len(build_flags) > 0 and bool(all(build_flags))
            else:
                # Flag arrays are removed after dataset finalisation
                # We therefore check missing dates against our expectation.
                expected_missing = set(ds_info.metadata.get("missing_dates", []))
                actual_missing = (
                    set(ds_info.dataset.missing)
                    if ds_info.dataset is not None
                    else None
                )
                if actual_missing is not None:
                    download_complete = actual_missing == expected_missing
                else:
                    download_complete = is_finalised
        except (AttributeError, FileNotFoundError, PathNotFoundError) as exc:
            msg = f"Unable to get status for {self.name} at {self.path_dataset}"
            raise RuntimeError(msg) from exc
        return AnemoiDatasetStatus(
            copy_in_progress=ds_info.copy_in_progress,
            download_complete=download_complete,
            is_finalised=is_finalised,
        )

    def create(self, *, overwrite: bool = False) -> None:
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
            try:
                status = self.check_status()
            except RuntimeError:
                logger.warning(
                    "Dataset %s at %s is in an unreadable state, likely from an interrupted "
                    "initialisation. Removing and re-downloading.",
                    self.name,
                    self.path_dataset,
                )
                shutil.rmtree(self.path_dataset, ignore_errors=True)
                self.download(overwrite=overwrite)
                return
            # The dataset is being downloaded
            if status.copy_in_progress:
                logger.warning(
                    "Dataset %s at %s is currently being downloaded by another process.",
                    self.name,
                    self.path_dataset,
                )
                return
            # If the download is complete then check whether the dataset is valid
            if status.download_complete:
                # Attempt to finalise, creating masks if necessary
                self.finalise(overwrite=overwrite, status=status)

                # Inspect the dataset for validity
                try:
                    self.inspect()
                    logger.info(
                        "Dataset %s at %s has been downloaded and seems to be valid.",
                        self.name,
                        self.path_dataset,
                    )
                except RuntimeError as exc:
                    # If the dataset is invalid we flag this to the user and exit
                    logger.error(  # noqa: TRY400
                        "Dataset %s at %s seems to be invalid. Please check manually.",
                        self.name,
                        self.path_dataset,
                    )
                    raise typer.Exit(1) from exc
                else:
                    # If the dataset is valid we return here
                    return

        # Download the dataset
        self.download(overwrite=overwrite)

    def download(self, *, overwrite: bool) -> None:
        """Download an Anemoi dataset in parts."""
        self.preprocessor.download(self.path_preprocessor)
        logger.info("Creating dataset %s at %s.", self.name, self.path_dataset)
        # Initialise
        self.initialise()
        # Load in parts
        self.load_in_chunks()
        # Attempt to finalise, creating masks if necessary
        status = self.check_status()
        if status.copy_in_progress:
            logger.warning(
                "Dataset %s at %s is still being copied by another process, skipping finalise.",
                self.name,
                self.path_dataset,
            )
        elif status.download_complete:
            self.finalise(overwrite=overwrite, status=status)
        else:
            logger.warning(
                "Dataset %s at %s is not fully loaded, skipping finalise.",
                self.name,
                self.path_dataset,
            )

    def finalise(self, *, overwrite: bool, status: AnemoiDatasetStatus) -> None:
        """Finalise the segmented Anemoi dataset."""
        if not status.is_finalised:
            Finalise().run(
                AnemoiFinaliseArgs(
                    path=str(self.path_dataset),
                    config=self.config,
                )
            )
            logger.info("Finalised dataset %s at %s.", self.name, self.path_dataset)

        # create active grid cell and land masks for the SSMIS dataset
        self.generate_masks(overwrite=overwrite)

    def generate_masks(self, *, overwrite: bool) -> None:
        """Generate land and active grid cell masks for the SSMIS dataset."""
        # if there is an SSMIS dataset, create the masks
        if "ssmis" not in self.name:
            logger.info("Not SSMIS dataset, skipping mask creation.")
            return

        self.path_masks.mkdir(parents=True, exist_ok=True)
        land_mask_path = self.path_masks / "land_mask.npy"
        active_mask_path = self.path_masks / "active_mask.npy"

        if land_mask_path.exists() and active_mask_path.exists() and not overwrite:
            logger.info("Both masks already exist, skipping creation.")
            return

        # Unpack status flags into a binary array
        ds_sf = open_dataset(self.path_dataset, select="status_flag")
        shape, dates = ds_sf.shape, ds_sf.dates
        status_flag = np.array(ds_sf).astype(np.uint8).reshape(*shape)
        binary = np.unpackbits(status_flag, axis=-1).reshape(*shape, 8)

        # land mask: land = 0, sea = 1
        if land_mask_path.exists() and not overwrite:
            logger.info("Land mask already exists, skipping creation.")
            land_mask = np.load(land_mask_path)
        else:
            land_mask = np.squeeze(binary[..., [7]]).sum(axis=0)
            # convert to binary mask
            land_mask = 1 - (land_mask > 0).astype(np.uint8)
            # reshape to 2D grid
            land_mask = land_mask.reshape(ds_sf.field_shape[-2:])
            # save the land mask for later use
            np.save(land_mask_path, land_mask)
            logger.info("Land mask created and saved.")

        # active mask: active grid cells = 1, inactive = 0
        if active_mask_path.exists() and not overwrite:
            logger.info("Active mask already exists, skipping creation.")
        else:
            # Identify grid cells that are inactive for all time steps
            inactive_mask = np.squeeze(binary[..., [0]]).sum(axis=0) >= dates.shape[0]
            # convert to binary mask, and set to 1 for active grid cells
            active_mask = 1 - (inactive_mask > 0).astype(np.uint8)
            # reshape to 2D grid
            active_mask = active_mask.reshape(ds_sf.field_shape[-2:])
            # intersect land mask with active mask to set all active grid cells to 1
            active_mask = active_mask * land_mask
            # save the active mask for later use
            np.save(active_mask_path, active_mask)
            logger.info("Active mask created and saved.")

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
        except (AttributeError, FileNotFoundError, PathNotFoundError) as exc:
            msg = f"Failed to initialise dataset {self.name} at {self.path_dataset}."
            logger.exception(msg)
            raise RuntimeError(msg) from exc

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
            try:
                InspectZarr().run(
                    AnemoiInspectArgs(
                        path=str(self.path_dataset),
                        detailed=detailed,
                        progress=(not detailed),
                        statistics=statistics,
                        size=size,
                    )
                )
            except (AttributeError, FileNotFoundError, PathNotFoundError) as exc:
                msg = f"Failed to load dataset {self.name} at {self.path_dataset}"
                raise RuntimeError(msg) from exc
        else:
            msg = f"Dataset {self.name} not found at {self.path_dataset}"
            raise RuntimeError(msg)

    def load_in_chunks(self) -> None:
        """Download a single Anemoi dataset in chunks, skipping those already present."""
        Load().run(
            AnemoiLoadArgs(
                path=str(self.path_dataset),
                config=self.config,
            )
        )
