import logging
from ftplib import FTP, error_perm
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import earthkit.data as ekd
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.create.sources.legacy import LegacySource
from anemoi.datasets.create.sources.xarray import load_one
from anemoi.datasets.dates.groups import GroupOfDates
from earthkit.data.core.fieldlist import FieldList, MultiFieldList
from earthkit.data.utils.patterns import Pattern

from icenet_mp.utils import to_list

logger = logging.getLogger(__name__)


@source_registry.register("ftp")
class FTPSource(LegacySource):
    @staticmethod
    def _execute(
        context: dict[str, Any],
        dates: GroupOfDates,
        url: str,
        passwd: str = "",
        user: str = "anonymous",
    ) -> ekd.FieldList:
        """Execute the data loading process from an FTP source."""
        # Parse the FTP URL
        server, path_pattern = url.replace("ftp://", "").split("/", 1)

        # Get list of remote file paths
        remote_paths = {
            date.isoformat(): to_list(
                Pattern(path_pattern).substitute(date=date, allow_extra=True)
            )
            for date in dates
        }

        # Connect to the FTP server
        downloaded_files: list[FieldList] = []
        with TemporaryDirectory() as tmpdir, FTP(server) as session:  # noqa: S321
            base_path = Path(tmpdir)
            session.login(user=user, passwd=passwd)

            # Iterate over remote paths
            for iso_date, remote_path_list in remote_paths.items():
                for remote_path in remote_path_list:
                    directory, filename = remote_path.rsplit("/", 1)
                    local_path = base_path / filename
                    try:
                        # Download the remote file
                        session.cwd(("/" + directory).replace("//", "/"))
                        with local_path.open("wb") as local_file:
                            session.retrbinary(f"RETR {filename}", local_file.write)
                        downloaded_files.append(
                            load_one("📂", context, [iso_date], str(local_path))
                        )
                    except error_perm as exc:
                        msg = f"Failed to download from '{remote_path}': {exc}"
                        logger.warning(msg)

        # Combine all downloaded files into a MultiFieldList
        return MultiFieldList(downloaded_files)
