import json
import logging
import os
import shutil
import socket
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep

from anemoi.datasets.commands.create import Create
from anemoi.datasets.commands.finalise import Finalise
from anemoi.datasets.commands.init import Init
from anemoi.datasets.commands.inspect import InspectZarr
from anemoi.datasets.commands.load import Load
from filelock import FileLock, Timeout
from omegaconf import DictConfig, OmegaConf
from zarr.errors import PathNotFoundError

from icenet_mp.types import (
    AnemoiCreateArgs,
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
            try:
                self.inspect()
                logger.info(
                    "Dataset %s already exists at %s, no need to download.",
                    self.name,
                    self.path_dataset,
                )
            except (AttributeError, FileNotFoundError, PathNotFoundError):
                # If the dataset is invalid we delete it
                logger.info("Dataset %s not found at %s.", self.name, self.path_dataset)
                shutil.rmtree(self.path_dataset, ignore_errors=True)
            else:
                # If the dataset is valid we return here
                return

        # Download the dataset
        self.download()

    def download(self) -> None:
        """Download a single Anemoi dataset."""
        self.preprocessor.download(self.path_preprocessor)
        logger.info("Creating dataset %s at %s.", self.name, self.path_dataset)
        Create().run(
            AnemoiCreateArgs(
                path=str(self.path_dataset),
                config=self.config,
            )
        )

    def inspect(self) -> None:
        """Inspect a single Anemoi dataset."""
        logger.info("Inspecting dataset %s at %s.", self.name, self.path_dataset)
        if self.path_dataset.exists():
            InspectZarr().run(
                AnemoiInspectArgs(
                    path=str(self.path_dataset),
                    detailed=True,
                    progress=True,
                    statistics=False,
                    size=True,
                )
            )
        else:
            logger.error("Dataset %s not found at %s.", self.name, self.path_dataset)

    def init(self, *, overwrite: bool) -> None:
        """Initialise a single Anemoi dataset."""
        logger.info("Initialising dataset %s at %s.", self.name, self.path_dataset)

        if overwrite:
            logger.info(
                "Overwrite set to true, reinitialising %s to %s",
                self.name,
                self.path_dataset,
            )
            shutil.rmtree(self.path_dataset, ignore_errors=True)
            Init().run(
                AnemoiInitArgs(
                    path=str(self.path_dataset),
                    config=self.config,
                )
            )
        else:
            try:
                self.inspect()
                logger.info(
                    "Dataset %s already exists at %s, no need to download.",
                    self.name,
                    self.path_dataset,
                )
            except (AttributeError, FileNotFoundError, PathNotFoundError):
                logger.info(
                    "Dataset %s not found at %s, initialising.",
                    self.name,
                    self.path_dataset,
                )
                shutil.rmtree(self.path_dataset, ignore_errors=True)
                Init().run(
                    AnemoiInitArgs(
                        path=str(self.path_dataset),
                        config=self.config,
                    )
                )

    def _part_tracker_path(self) -> Path:
        """Path for part_trackerdata file that tracks completed parts."""
        return (
            Path(self.path_dataset).with_suffix(".load_parts.json")
            if not Path(self.path_dataset).is_dir()
            else Path(self.path_dataset) / ".load_parts.json"
        )

    def _part_tracker_lock_path(self) -> Path:
        """Path for the lockfile used when updating the part tracker."""
        return self._part_tracker_path().with_suffix(".lock")

    def _read_part_tracker(self) -> dict:
        """Read part_tracker data JSON (returns {'completed': {part_spec: timestamp}})."""
        part_tracker_file = self._part_tracker_path()
        if not part_tracker_file.exists():
            return {"completed": {}}
        try:
            with part_tracker_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Failed to read load chunks part_tracker data %s; starting fresh",
                part_tracker_file,
            )
            return {"completed": {}}

    def _write_part_tracker(self, data: dict) -> None:
        """Write part_tracker data JSON to disk.

        Ensure parent dir exists. We write to a temporary file in the same directory
        then use replace() to get an atomic move. This is a POSIX-only feature.
        This means that if the process is killed, the part_tracker data will not be lost.
        """
        part_tracker_file = self._part_tracker_path()
        # Ensure parent directory exists (NamedTemporaryFile requires it)
        try:
            part_tracker_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.debug(
                "Could not ensure parent dir for part_tracker data: %s",
                part_tracker_file.parent,
            )

        try:
            with NamedTemporaryFile(
                "w", delete=False, dir=str(part_tracker_file.parent), encoding="utf-8"
            ) as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
                tmp_name = Path(fh.name)
            tmp_name.replace(part_tracker_file)  # atomic replace if on same filesystem
        except OSError:
            # fallback to naive write
            try:
                with part_tracker_file.open("w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, sort_keys=True)
            except OSError:
                logger.exception(
                    "Failed to write part_tracker data file %s", part_tracker_file
                )

    def load(self, parts: str) -> None:
        """Download a segment of an Anemoi dataset."""
        logger.info(
            "Downloading %s part of %s to %s.",
            parts,
            self.name,
            self.path_dataset,
        )
        Load().run(
            AnemoiLoadArgs(
                path=str(self.path_dataset),
                config=self.config,
                parts=parts,
            )
        )

    def load_in_parts(  # noqa: C901, PLR0915
        self,
        *,
        continue_on_error: bool = True,
        force_reset: bool = False,
        use_lock: bool = True,
        lock_timeout: int = 30,
        total_parts: int = 10,
        overwrite: bool = False,
    ) -> None:
        """Load all parts automatically and record progress so runs can be resumed.

        Args:
            continue_on_error: if True, log errors and continue with later parts.
            force_reset: if True, clear part_tracker file and re-run all parts from scratch.
            use_lock: if True, use file locking to prevent concurrent updates to part_tracker file.
            lock_timeout: timeout in seconds for acquiring the lock.
            total_parts: number of parts to load, default = 10.
            overwrite: if True, delete the dataset directory before loading.

        """

        # Nested helper functions that access outer scope variables
        def check_and_mark_in_progress(part_spec: str) -> bool:
            """Check if part should be skipped and mark it as in progress.

            Returns True if part should be skipped (already completed), False otherwise.
            """
            lock_path = self._part_tracker_lock_path()
            max_retries = 3
            backoff = 1.0  # seconds between lock retries

            def _claim(part_tracker: dict) -> bool:
                completed = set(part_tracker.get("completed", {}).keys())
                # Skip if already done
                # if part_spec in completed:
                if part_spec in completed:
                    logger.info(
                        "Skipping already completed part %s for %s",
                        part_spec,
                        self.name,
                    )
                    return True

                # mark in-progress with PID/host
                part_tracker.setdefault("in_progress", {})[part_spec] = {
                    "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                }
                self._write_part_tracker(part_tracker)
                return False

            if use_lock:
                for attempt in range(1, max_retries + 1):
                    try:
                        with FileLock(str(lock_path), timeout=lock_timeout):
                            part_tracker = self._read_part_tracker()
                            return _claim(part_tracker)
                    except Timeout:
                        logger.warning(
                            "Timeout acquiring part-tracker lock for %s (attempt %d/%d)",
                            self.name,
                            attempt,
                            max_retries,
                        )
                        if attempt < max_retries:
                            sleep(backoff)
                            backoff *= 2
                            continue
                        # final attempt failed — skip (will retry next run)
                        logger.exception(
                            "Failed to acquire lock after %d attempts, skipping part %s",
                            max_retries,
                            part_spec,
                        )
                        return True
            else:
                part_tracker = self._read_part_tracker()
                return _claim(part_tracker)
            return False  # Explicit return for type checker

        def clear_part_in_progress(part_spec: str) -> None:
            """Clear the in_progress status for a part."""
            lock_path = self._part_tracker_lock_path()
            if use_lock:
                try:
                    with FileLock(str(lock_path), timeout=lock_timeout):
                        part_tracker = self._read_part_tracker()
                        part_tracker.get("in_progress", {}).pop(part_spec, None)
                        self._write_part_tracker(part_tracker)
                except Timeout:
                    logger.warning(
                        "Timeout acquiring lock to clear in_progress for %s", part_spec
                    )
            else:
                part_tracker = self._read_part_tracker()
                part_tracker.get("in_progress", {}).pop(part_spec, None)
                self._write_part_tracker(part_tracker)

        def mark_part_completed(part_spec: str) -> None:
            """Mark a part as completed in the tracker."""
            ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            lock_path = self._part_tracker_lock_path()
            if use_lock:
                try:
                    with FileLock(str(lock_path), timeout=lock_timeout):
                        part_tracker = self._read_part_tracker()
                        part_tracker.get("in_progress", {}).pop(part_spec, None)
                        part_tracker.setdefault("completed", {})[part_spec] = {
                            "completed_at": ts,
                            "pid": os.getpid(),
                            "host": socket.gethostname(),
                        }
                        self._write_part_tracker(part_tracker)
                except Timeout:
                    logger.warning(
                        "Timeout acquiring lock to mark part %s completed; data likely written",
                        part_spec,
                    )
            else:
                part_tracker = self._read_part_tracker()
                part_tracker.get("in_progress", {}).pop(part_spec, None)
                part_tracker.setdefault("completed", {})[part_spec] = {
                    "completed_at": ts,
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                }
                self._write_part_tracker(part_tracker)

        if "sic" in self.name:
            logger.warning(
                "Loading dataset %s in parts is not supported for non-anemoi datasets, use create.",
                self.name,
            )
            return

        if overwrite:
            logger.info(
                "overwrite set to true, deleting dataset %s at %s",
                self.name,
                self.path_dataset,
            )
            shutil.rmtree(self.path_dataset, ignore_errors=True)
            # Also delete the tracker file if it exists (may be outside dataset directory)
            tracker_path = self._part_tracker_path()
            if tracker_path.exists():
                tracker_path.unlink()
                logger.info("Deleted part tracker file at %s", tracker_path)
            # Initialize the dataset after deletion so we can load parts into it
            self.init(overwrite=False)
        else:
            # Ensure dataset is initialized before loading parts
            try:
                self.inspect()

                if not self._part_tracker_path().exists():
                    logger.info(
                        "Dataset %s already exists at %s.", self.name, self.path_dataset
                    )
                    return
            except (AttributeError, FileNotFoundError, PathNotFoundError):
                logger.info(
                    "Dataset %s not found at %s, initialising before loading parts.",
                    self.name,
                    self.path_dataset,
                )
                self.init(overwrite=False)

        logger.info(
            "Starting chunked load for dataset %s (parts=%d)", self.name, total_parts
        )

        part_tracker = self._read_part_tracker()
        # Skip force_reset if overwrite was used (already cleared everything)
        if (
            force_reset
            and not overwrite
            and (part_tracker.get("completed") or part_tracker.get("in_progress"))
        ):
            logger.info(
                "force_reset requested — clearing previous load part_tracker data (completed + in_progress) at %s",
                self._part_tracker_path(),
            )
            part_tracker = {"completed": {}}
            # ensure no in_progress remains
            self._write_part_tracker(part_tracker)

        for i in range(1, total_parts + 1):
            part_spec = f"{i}/{total_parts}"
            # Check if part should be skipped and mark as in progress
            if check_and_mark_in_progress(part_spec):
                continue

            # Do the heavy work outside the lock
            logger.info("Loading part %s for dataset %s", part_spec, self.name)
            try:
                self.load(parts=part_spec)
            except Exception:
                logger.exception(
                    "Error while loading part %s for dataset %s", part_spec, self.name
                )
                # clear in_progress under lock and optionally re-raise
                clear_part_in_progress(part_spec)

                if not continue_on_error:
                    raise
                logger.info("Continuing to next part after error in %s", part_spec)
                continue

            # Mark completed under lock
            mark_part_completed(part_spec)
            logger.info(
                "Marked part %s as completed for dataset %s", part_spec, self.name
            )

        logger.info("Chunked load finished for dataset %s", self.name)

    def finalise(self) -> None:
        """Finalise the segmented Anemoi dataset."""
        Finalise().run(
            AnemoiFinaliseArgs(
                path=str(self.path_dataset),
                config=self.config,
            )
        )
