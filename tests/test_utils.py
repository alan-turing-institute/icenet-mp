import re
from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from lightning import Trainer

from icenet_mp.utils import (
    datetime_from_npdatetime,
    get_device_name,
    get_timestamp,
    get_wandb_run,
    mask_dir,
    normalise_date,
    to_list,
)


class TestDatetimeFromNpdatetime:
    def test_returns_utc_datetime(self) -> None:
        """Convert NumPy datetimes to timezone-aware UTC datetimes."""
        result = datetime_from_npdatetime(np.datetime64("2026-08-21T12:34:56.789"))

        assert result.tzinfo is UTC
        assert result.microsecond == 789000


class TestMaskDir:
    def test_builds_expected_path(self, tmp_path: Path) -> None:
        """Build the preprocessing mask directory beneath the configured root."""
        assert mask_dir(tmp_path, "sic-ssmis") == (
            tmp_path / "data/preprocessing/masks/sic-ssmis"
        )


class TestGetDeviceName:
    @pytest.mark.parametrize(
        ("accelerator_name", "expected"),
        [("cpu", "CPU"), ("mps", "Apple Silicon GPU"), ("auto", "CPU")],
    )
    def test_for_non_cuda_accelerators(
        self, accelerator_name: str, expected: str
    ) -> None:
        """Return readable names for non-CUDA accelerator selections."""
        assert get_device_name(accelerator_name) == expected

    def test_handles_unavailable_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fall back cleanly when the CUDA device name cannot be queried."""

        def _raise_assertion() -> str:
            raise AssertionError

        monkeypatch.setattr(torch.cuda, "get_device_name", _raise_assertion)

        assert get_device_name("cuda") == "Unknown CUDA device"


class TestGetTimestamp:
    def test_has_expected_utc_format(self) -> None:
        """Format generated timestamps using the expected UTC pattern."""
        assert re.fullmatch(r"\d{8}_\d{6}", get_timestamp())


class TestGetWandbRun:
    def test_returns_none_without_wandb_logger(self) -> None:
        """Return no W&B run when the trainer has no W&B logger."""
        trainer = MagicMock(spec=Trainer)
        trainer.loggers = [object()]

        assert get_wandb_run(trainer) is None


class TestNormaliseDate:
    def test_sets_time_to_noon(self) -> None:
        """Normalize dates to noon while preserving the calendar date."""
        result = normalise_date(np.datetime64("2026-08-21T03:14:15.926"))

        assert result == np.datetime64("2026-08-21T12:00:00.000")


class TestToList:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("ice_conc", ["ice_conc"]), (["ice_conc", "2t"], ["ice_conc", "2t"])],
        ids=["scalar", "list"],
    )
    def test_to_list(self, value: str | list[str], expected: list[str]) -> None:
        """Normalize scalar strings and string lists to list form."""
        assert to_list(value) == expected
