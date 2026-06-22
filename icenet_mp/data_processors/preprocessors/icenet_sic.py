import logging
import os
import shutil
from collections.abc import MutableMapping
from ftplib import FTP, error_perm
from pathlib import Path
from typing import Any

import pandas as pd
from icenet.data.sic.mask import Masks
from icenet.data.sic.osisaf import SICDownloader

from .ipreprocessor import IPreprocessor

logger = logging.getLogger(__name__)


class IceNetSICPreprocessor(IPreprocessor):
    def __init__(self, config: MutableMapping[str, Any]) -> None:
        """Initialise the IceNetSIC preprocessor."""
        super().__init__(config)
        self.date_range = pd.date_range(
            config["dates"]["start"],
            config["dates"]["end"],
            freq=config["dates"]["frequency"],
        )
        self.is_north = config["preprocessor"]["hemisphere"].lower() == "north"

    @property
    def is_south(self) -> bool:
        return not self.is_north

    def download(self, preprocessor_path: Path) -> None:
        """Download data to the specified preprocessor path."""
        # Change to the output directory before downloading
        icenet_path = preprocessor_path / self.dataset_name / self.cls_name
        icenet_path.mkdir(parents=True, exist_ok=True)
        current_directory = Path.cwd()
        os.chdir(icenet_path)
        logger.info("OSISAF SIC data will be downloaded to %s.", icenet_path)

        # Treat each year separately
        for year in sorted({date.year for date in self.date_range}):
            logger.info("Preparing to download data for %s.", year)
            # Generate polar masks: note that only 1979-2015 are available
            masks = Masks(north=self.is_north, south=self.is_south)
            mask_year = max(1979, min(year, 2015))
            logger.info("Downloading polar masks for %04d.", mask_year)
            self.pre_download_masks(mask_year, icenet_path)
            masks.generate(year=mask_year)

            logger.info("Downloading sea ice concentration data for %04d.", year)
            sic = SICDownloader(
                dates=[
                    pd.to_datetime(date).date()
                    for date in self.date_range
                    if date.year == year
                ],  # Dates to download SIC data for
                delete_tempfiles=True,  # Delete temporary downloaded files after use
                north=self.is_north,  # Use mask for the Northern Hemisphere (set to True if needed)
                south=self.is_south,  # Use mask for the Southern Hemisphere
                parallel_opens=False,  # Concatenation is not working correctly with dask
            )
            sic.download()

        # Change back to the original directory
        os.chdir(current_directory)

    def pre_download_masks(self, mask_year: int, icenet_path: Path) -> None:
        """Pre-download the OSISAF masks for a given year.

        We need to do this ourselves, as IceNet tries to download the 2nd day of each
        month, which does not always exist.
        """
        mask_base_path = (
            icenet_path
            / "data"
            / "masks"
            / ("north" if self.is_north else "south")
            / "siconca"
        )
        last_filepath: None | Path = None
        ftp_client = FTP("osisaf.met.no", "anonymous")  # noqa: S321
        for mask_month in range(1, 13):
            # Create the directory we will download to
            filename_stem = f"ice_conc_{'nh' if self.is_north else 'sh'}_ease2-250_cdr-v2p0_{mask_year:04d}{mask_month:02d}"
            local_filepath = (
                mask_base_path
                / f"{mask_year:04d}"
                / f"{mask_month:02d}"
                / f"{filename_stem}021200.nc"
            )
            local_filepath.parent.mkdir(parents=True, exist_ok=True)
            # Try to change to the directory for this month, copying the last month if
            # it does not exist
            try:
                ftp_client.cwd(
                    f"reprocessed/ice/conc/v2p0/{mask_year:04d}/{mask_month:02d}"
                )
            except error_perm as e:
                if last_filepath is None:
                    msg = f"No mask found for {mask_year}-{mask_month}, and no previous month to copy from."
                    raise RuntimeError(msg) from e
                logger.warning(
                    "No mask found for %02d-%02d, using previous month.",
                    mask_year,
                    mask_month,
                )
                shutil.copyfile(last_filepath, local_filepath)
                continue
            # Try to download for each day of the month until we find one that exists
            for mask_day in range(1, 29):
                if not (local_filepath.is_file() and local_filepath.stat().st_size):
                    with local_filepath.open("wb") as fp:
                        try:
                            remote_filename = f"{filename_stem}{mask_day:02d}1200.nc"
                            ftp_client.retrbinary(f"RETR {remote_filename}", fp.write)
                            logger.info(
                                "Using mask from day %s for %02d-%02d.",
                                mask_day,
                                mask_year,
                                mask_month,
                            )
                        except error_perm:
                            logger.warning(
                                "Mask for day %s does not exist for %02d-%02d.",
                                mask_day,
                                mask_year,
                                mask_month,
                            )
                else:
                    last_filepath = local_filepath
