import re
from datetime import UTC
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from wandb.wandb_run import Run

from icenet_mp.utils import (
    datetime_from_npdatetime,
    get_device_name,
    get_timestamp,
    get_wandb_run,
    mask_dir,
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

    @pytest.mark.parametrize(
        ("accelerator_name", "torch_module"),
        [("cuda", torch.cuda), ("xpu", torch.xpu)],
        ids=["cuda", "xpu"],
    )
    def test_returns_the_queried_device_name(
        self,
        accelerator_name: str,
        torch_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return the accelerator's own reported name when the query succeeds."""
        monkeypatch.setattr(torch_module, "get_device_name", lambda: "Custom GPU")

        assert get_device_name(accelerator_name) == "Custom GPU"

    @pytest.mark.parametrize(
        ("accelerator_name", "torch_module", "expected"),
        [
            ("cuda", torch.cuda, "Unknown CUDA device"),
            ("xpu", torch.xpu, "Unknown XPU device"),
        ],
        ids=["cuda", "xpu"],
    )
    def test_handles_unavailable_accelerator(
        self,
        accelerator_name: str,
        torch_module: ModuleType,
        expected: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fall back cleanly when the accelerator's device name cannot be queried."""

        def _raise_assertion() -> str:
            raise AssertionError

        monkeypatch.setattr(torch_module, "get_device_name", _raise_assertion)

        assert get_device_name(accelerator_name) == expected


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

    def test_returns_the_run_from_a_wandb_logger(self) -> None:
        """Return the active W&B run when a WandbLogger is present."""
        run = MagicMock(spec=Run)
        wandb_logger = MagicMock(spec=WandbLogger)
        wandb_logger.experiment = run
        trainer = MagicMock(spec=Trainer)
        trainer.loggers = [wandb_logger]

        assert get_wandb_run(trainer) is run

    def test_returns_none_when_wandb_logger_has_no_run_experiment(self) -> None:
        """Return None when the WandbLogger's experiment isn't a Run (e.g. offline)."""
        wandb_logger = MagicMock(spec=WandbLogger)
        wandb_logger.experiment = None
        trainer = MagicMock(spec=Trainer)
        trainer.loggers = [wandb_logger]

        assert get_wandb_run(trainer) is None


class TestToList:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("ice_conc", ["ice_conc"]), (["ice_conc", "2t"], ["ice_conc", "2t"])],
        ids=["scalar", "list"],
    )
    def test_to_list(self, value: str | list[str], expected: list[str]) -> None:
        """Normalize scalar strings and string lists to list form."""
        assert to_list(value) == expected
