import re
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icenet_mp.utils import (
    datetime_from_npdatetime,
    get_device_name,
    get_timestamp,
    get_wandb_run,
    mask_dir,
    normalise_date,
    to_list,
)


def test_datetime_from_npdatetime_returns_utc_datetime() -> None:
    """Convert NumPy datetimes to timezone-aware UTC datetimes."""
    result = datetime_from_npdatetime(np.datetime64("2026-08-21T12:34:56.789"))

    assert result.tzinfo is UTC
    assert result.microsecond == 789000


def test_mask_dir_builds_expected_path(tmp_path: Path) -> None:
    """Build the preprocessing mask directory beneath the configured root."""
    assert mask_dir(tmp_path, "sic-ssmis") == (
        tmp_path / "data/preprocessing/masks/sic-ssmis"
    )


@pytest.mark.parametrize(
    ("accelerator_name", "expected"),
    [("cpu", "CPU"), ("mps", "Apple Silicon GPU"), ("auto", "CPU")],
)
def test_get_device_name_for_non_cuda_accelerators(
    accelerator_name: str, expected: str
) -> None:
    """Return readable names for non-CUDA accelerator selections."""
    assert get_device_name(accelerator_name) == expected


def test_get_device_name_handles_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back cleanly when the CUDA device name cannot be queried."""

    def _raise_assertion() -> str:
        raise AssertionError

    monkeypatch.setattr(torch.cuda, "get_device_name", _raise_assertion)

    assert get_device_name("cuda") == "Unknown CUDA device"


def test_get_timestamp_has_expected_utc_format() -> None:
    """Format generated timestamps using the expected UTC pattern."""
    assert re.fullmatch(r"\d{8}_\d{6}", get_timestamp())


def test_get_wandb_run_returns_none_without_wandb_logger() -> None:
    """Return no W&B run when the trainer has no W&B logger."""
    trainer = SimpleNamespace(loggers=[object()])

    assert get_wandb_run(trainer) is None  # type: ignore[arg-type]


def test_normalise_date_sets_time_to_noon() -> None:
    """Normalize dates to noon while preserving the calendar date."""
    result = normalise_date(np.datetime64("2026-08-21T03:14:15.926"))

    assert result == np.datetime64("2026-08-21T12:00:00.000")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("ice_conc", ["ice_conc"]), (["ice_conc", "2t"], ["ice_conc", "2t"])],
)
def test_to_list(value: str | list[str], expected: list[str]) -> None:
    """Normalize scalar strings and string lists to list form."""
    assert to_list(value) == expected
