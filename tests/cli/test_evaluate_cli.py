from pathlib import Path

import pytest
from omegaconf import DictConfig
from typer.testing import CliRunner

from icenet_mp.cli.main import app
from icenet_mp.model_service import ModelService


class FakeModelService:
    def __init__(self) -> None:
        """Initialise evaluation-call tracking."""
        self.evaluate_calls = 0

    def evaluate(self) -> None:
        self.evaluate_calls += 1


class TestEvaluateCLI:
    def test_evaluate_loads_resolved_checkpoint_and_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Compose the config, resolve the checkpoint path, and run evaluate()."""
        service = FakeModelService()
        captured: list[tuple[DictConfig, Path]] = []

        def fake_from_checkpoint(
            config: DictConfig, checkpoint: Path
        ) -> FakeModelService:
            captured.append((config, checkpoint))
            return service

        monkeypatch.setattr(ModelService, "from_checkpoint", fake_from_checkpoint)
        checkpoint = tmp_path / "model.ckpt"

        result = CliRunner().invoke(
            app,
            [
                "evaluate",
                "--config-name",
                "sample",
                "--checkpoint",
                str(checkpoint),
            ],
            prog_name="imp",
        )

        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        assert captured[0][1] == checkpoint.resolve()
        assert captured[0][0].model.name == "quick-test"
        assert (
            list(captured[0][0].evaluate.callbacks.activation_saver.layer_paths) == []
        )
        assert service.evaluate_calls == 1

    def test_checkpoint_resolves_a_relative_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relative --checkpoint is resolved against the current directory."""
        service = FakeModelService()
        captured: list[tuple[DictConfig, Path]] = []

        def fake_from_checkpoint(
            config: DictConfig, checkpoint: Path
        ) -> FakeModelService:
            captured.append((config, checkpoint))
            return service

        monkeypatch.setattr(ModelService, "from_checkpoint", fake_from_checkpoint)
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(
            app,
            [
                "evaluate",
                "--config-name",
                "sample",
                "--checkpoint",
                "model.ckpt",
            ],
            prog_name="imp",
        )

        assert result.exit_code == 0, result.output
        assert captured[0][1] == (tmp_path / "model.ckpt").resolve()

    def test_repeated_save_layer_flags_update_activation_saver_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeat --save-layer to hook multiple submodules in one run."""
        service = FakeModelService()
        captured: list[DictConfig] = []

        def fake_from_checkpoint(
            config: DictConfig, _checkpoint: Path
        ) -> FakeModelService:
            captured.append(config)
            return service

        monkeypatch.setattr(ModelService, "from_checkpoint", fake_from_checkpoint)

        result = CliRunner().invoke(
            app,
            [
                "evaluate",
                "--config-name",
                "sample",
                "--checkpoint",
                str(tmp_path / "model.ckpt"),
                "--save-layer",
                "processor.conv1",
                "--save-layer",
                "decoder.model",
            ],
            prog_name="imp",
        )

        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        assert list(captured[0].evaluate.callbacks.activation_saver.layer_paths) == [
            "processor.conv1",
            "decoder.model",
        ]
        assert service.evaluate_calls == 1
