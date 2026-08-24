import re
from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from icenet_mp import utils


def test_datetime_from_npdatetime_returns_utc_datetime() -> None:
    """Convert NumPy datetimes to timezone-aware UTC datetimes."""
    result = utils.datetime_from_npdatetime(np.datetime64("2026-08-24T17:30:45.123"))

    assert result.tzinfo is UTC
    assert result.isoformat() == "2026-08-24T17:30:45.123000+00:00"


def test_mask_dir_uses_shared_preprocessing_layout(tmp_path: Path) -> None:
    """Build mask paths under the shared preprocessing directory."""
    assert utils.mask_dir(tmp_path, "sic-osisaf") == (
        tmp_path / "data" / "preprocessing" / "masks" / "sic-osisaf"
    )


@pytest.mark.parametrize(
    ("accelerator", "expected"),
    [("cpu", "CPU"), ("auto", "CPU"), ("mps", "Apple Silicon GPU")],
)
def test_get_device_name_for_non_discrete_accelerators(
    accelerator: str, expected: str
) -> None:
    """Return stable labels for CPU, auto and Apple Silicon accelerators."""
    assert utils.get_device_name(accelerator) == expected


def test_get_device_name_uses_cuda_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read the CUDA device name from the runtime when available."""
    monkeypatch.setattr(utils.torch.cuda, "get_device_name", lambda: "Test CUDA GPU")

    assert utils.get_device_name("cuda") == "Test CUDA GPU"


def test_get_device_name_handles_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back cleanly when CUDA device discovery is unavailable."""

    def unavailable() -> str:
        raise AssertionError

    monkeypatch.setattr(utils.torch.cuda, "get_device_name", unavailable)

    assert utils.get_device_name("cuda") == "Unknown CUDA device"


def test_get_device_name_uses_xpu_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read the XPU device name from the runtime when available."""
    xpu = MagicMock()
    xpu.get_device_name.return_value = "Test XPU"
    monkeypatch.setattr(utils.torch, "xpu", xpu)

    assert utils.get_device_name("xpu") == "Test XPU"


def test_get_device_name_handles_unavailable_xpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back cleanly when XPU device discovery is unavailable."""
    xpu = MagicMock()
    xpu.get_device_name.side_effect = AssertionError
    monkeypatch.setattr(utils.torch, "xpu", xpu)

    assert utils.get_device_name("xpu") == "Unknown XPU device"


def test_get_timestamp_has_expected_utc_shape() -> None:
    """Format timestamps using the expected compact UTC shape."""
    assert re.fullmatch(r"\d{8}_\d{6}", utils.get_timestamp()) is not None


def test_get_wandb_run_returns_first_real_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the first matching WandB run from trainer loggers."""

    class FakeRun:
        pass

    class FakeWandbLogger:
        def __init__(self, experiment: object) -> None:
            self.experiment = experiment

    monkeypatch.setattr(utils, "Run", FakeRun)
    monkeypatch.setattr(utils, "WandbLogger", FakeWandbLogger)

    run = FakeRun()
    trainer = MagicMock()
    trainer.loggers = [object(), FakeWandbLogger(run), FakeWandbLogger(FakeRun())]

    assert utils.get_wandb_run(trainer) is run


def test_get_wandb_run_returns_none_without_matching_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return None when no trainer logger exposes a matching WandB run."""

    class FakeRun:
        pass

    class FakeWandbLogger:
        def __init__(self, experiment: object) -> None:
            self.experiment = experiment

    monkeypatch.setattr(utils, "Run", FakeRun)
    monkeypatch.setattr(utils, "WandbLogger", FakeWandbLogger)

    trainer = MagicMock()
    trainer.loggers = [object(), FakeWandbLogger(object())]

    assert utils.get_wandb_run(trainer) is None


def test_normalise_date_moves_time_to_noon() -> None:
    """Normalise arbitrary timestamps to noon on the same date."""
    result = utils.normalise_date(np.datetime64("2026-08-24T23:59:59"))

    assert result == np.datetime64("2026-08-24T12:00:00")


def test_to_list_wraps_string_and_preserves_list() -> None:
    """Wrap strings in a list while preserving existing list objects."""
    values = ["a", "b"]

    assert utils.to_list("a") == ["a"]
    assert utils.to_list(values) is values
