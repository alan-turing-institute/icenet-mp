import logging
import shutil
from pathlib import Path

import numpy as np
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
            copy_in_progress = ds_info.copy_in_progress
            statistics_ready = ds_info.statistics_ready
            if ds_info.dataset is None:
                # ... if there is no dataset object then the download is incomplete
                download_complete = False
            elif copy_in_progress:
                # If a copy is in progress then the download is incomplete
                download_complete = False
            elif (build_flags := ds_info.build_flags) is not None:
                # If build flags are present and all true then the download is complete
                download_complete = len(build_flags) > 0 and bool(all(build_flags))
            elif ds_info.statistics_started is not None:
                # If statistics generation has started then the download is complete
                download_complete = True
            else:
                msg = (
                    f"Unable to determine readiness status for dataset {self.name} at "
                    f"{self.path_dataset}. Please check manually."
                )
                raise RuntimeError(msg)
        except (AttributeError, FileNotFoundError, PathNotFoundError) as exc:
            msg = f"Unable to get status for {self.name} at {self.path_dataset}"
            raise RuntimeError(msg) from exc
        return AnemoiDatasetStatus(
            copy_in_progress=copy_in_progress,
            download_complete=download_complete,
            is_finalised=download_complete and statistics_ready,
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

        # Otherwise check whether a valid dataset exists
        elif self.path_dataset.exists():
            try:
                status = self.check_status()
            except RuntimeError as exc:
                msg = f"Status of dataset {self.name} at {self.path_dataset} could not be determined. Please check manually."
                logger.error(msg)  # noqa: TRY400
                raise RuntimeError(msg) from exc
            else:
                if status.copy_in_progress:
                    # If the dataset is being written to, we exit without error
                    logger.warning(
                        "Dataset %s at %s is currently being downloaded by another process.",
                        self.name,
                        self.path_dataset,
                    )
                    return
                if status.download_complete:
                    self.finalise(overwrite=overwrite, status=status)
                    try:
                        self.inspect()
                    except RuntimeError as exc:
                        msg = f"Dataset {self.name} at {self.path_dataset} could not be inspected. Please check manually."
                        logger.error(msg)  # noqa: TRY400
                        raise RuntimeError(msg) from exc
                    # At this point we have a valid dataset so we exit without error
                    logger.info(
                        "Dataset %s at %s has been downloaded and seems to be valid.",
                        self.name,
                        self.path_dataset,
                    )
                    return

        # At this point there is no dataset at the requested path so we download to it
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
            logger.error(msg)  # noqa: TRY400
            raise RuntimeError(msg) from exc

    def inspect(
        self,
        *,
        verbose: bool = False,
    ) -> None:
        """Inspect an Anemoi dataset."""
        if self.path_dataset.exists():
            try:
                if verbose:
                    InspectZarr().run(
                        AnemoiInspectArgs(
                            path=str(self.path_dataset),
                            detailed=True,
                            progress=False,
                            statistics=False,  # recalculate statistics on-the-fly
                            size=True,
                        )
                    )
                else:
                    ds_info = InspectZarr()._info(str(self.path_dataset))
                    ds_info.describe()
                    ds_info.progress()
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
