from pathlib import Path

import pytest
from omegaconf import DictConfig
from typer.testing import CliRunner

from icenet_mp.cli.main import app
from icenet_mp.model_service import ModelService


class FakeModelService:
    def __init__(self) -> None:
        self.calls: list[tuple[Path | None, bool]] = []

    def train(
        self, checkpoint_dir: Path | None = None, *, multistage: bool = False
    ) -> None:
        self.calls.append((checkpoint_dir, multistage))


class TestTrainCLI:
    def test_default_training_forwards_composed_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FakeModelService()
        captured: list[DictConfig] = []

        def fake_from_config(config: DictConfig) -> FakeModelService:
            captured.append(config)
            return service

        monkeypatch.setattr(ModelService, "from_config", fake_from_config)

        result = CliRunner().invoke(
            app,
            ["train", "--config-name", "sample"],
            prog_name="imp",
        )

        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        assert captured[0].model.name == "quick-test"
        assert service.calls == [(None, False)]

    def test_multistage_training_resolves_checkpoint_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FakeModelService()

        def fake_from_config(_config: DictConfig) -> FakeModelService:
            return service

        monkeypatch.setattr(ModelService, "from_config", fake_from_config)
        checkpoint_dir = tmp_path / "checkpoints"

        result = CliRunner().invoke(
            app,
            [
                "train",
                "--config-name",
                "sample",
                "--multistage",
                "--checkpoint-dir",
                str(checkpoint_dir),
            ],
            prog_name="imp",
        )

        assert result.exit_code == 0, result.output
        assert service.calls == [(checkpoint_dir.resolve(), True)]
