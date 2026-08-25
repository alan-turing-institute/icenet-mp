import logging
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from optuna.trial import TrialState
from typer.testing import Result

from icenet_mp.model_service import ModelService
from icenet_mp.sweep import OptunaSweep

from .conftest import CustomCliRunner


class FakeTrainer:
    def __init__(self, checkpoint_callbacks: list[object]) -> None:
        """A fake Trainer exposing only the checkpoint_callbacks the CLI reads."""
        self.checkpoint_callbacks = checkpoint_callbacks


class FakeModelService:
    def __init__(self, trainer: FakeTrainer) -> None:
        """A fake ModelService that returns a fixed FakeTrainer from train()."""
        self._trainer = trainer

    def train(self, *, checkpoint_dir: Path | None, multistage: bool) -> FakeTrainer:  # noqa: ARG002
        return self._trainer


class TestSweepCLI:
    @staticmethod
    def _build_study(tmp_path: Path, n_completed: int = 1) -> tuple[Path, int | None]:
        """Create a local sweep directory with `n_completed` completed trials.

        Returns the sweep directory and the number of the last (so also best, since
        each trial scores 0.5) completed trial, or None if `n_completed` is 0.
        """
        cfg_sweep = {
            "name": "example",
            "n_trials": 3,
            "sampler": "random",
            "seed": 0,
            "entity": "test-entity",
            "parameters": {
                "train.optimizer.lr": {"type": "float", "low": 1.0e-5, "high": 1.0e-2}
            },
        }
        study_path = tmp_path / "example-sweep"
        study_path.mkdir()
        (study_path / "optuna.yaml").write_text(yaml.safe_dump(cfg_sweep))
        OmegaConf.save(
            OmegaConf.create({"train": {"optimizer": {"lr": 0.001}}}),
            study_path / "model_config.yaml",
        )

        sampler = OptunaSweep.from_path(study_path)
        trial_number = None
        for _ in range(n_completed):
            trial, _ = sampler.ask()
            sampler.tell(trial, 0.5)
            trial_number = trial.number
        return study_path, trial_number

    def test_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["sweep", "--help"],
            expected_patterns=[
                r"Usage: imp sweep \[OPTIONS\] COMMAND \[ARGS\]...",
                r"Generate W&B sweeps with Optuna-sampled hyperparameters",
                r"--help\s+-h\s+Show this message and exit.",
                r"initialise\s+Initialise a W&B sweep with Optuna-sampled",
                r"summarise\s+Summarise the best parameters found in a",
                r"trial\s+Run a single trial from a W&B sweep.",
            ],
        )

    def test_initialise_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["sweep", "initialise", "--help"],
            expected_patterns=[
                r"Usage: imp sweep initialise \[OPTIONS\] \[overrides\]...",
                r"Initialise a W&B sweep with Optuna-sampled hyperparameters.",
                r"overrides\s+<str>\s+One or more space-separated Hydra config overrides",
                r"--sweep-yaml\s+<path>\s+Full path to a sweep search-space YAML",
                r"--config-name\s+<str>\s+Name of a file to load from the config",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )

    def test_summarise_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["sweep", "summarise", "--help"],
            expected_patterns=[
                r"Usage: imp sweep summarise \[OPTIONS\]",
                r"Summarise the best parameters found in a W&B sweep.",
                r"--sweep-path\s+<path>\s+Full path to a local sweep directory",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )

    def test_trial_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["sweep", "trial", "--help"],
            expected_patterns=[
                r"Usage: imp sweep trial \[OPTIONS\]",
                r"Run a single trial from a W&B sweep.",
                r"--sweep-path\s+<path>\s+Full path to a local sweep directory",
                r"--checkpoint-dir\s+<str>\s+Path to a directory of existing",
                r"--multistage\s+Train an EncodeProcessDecode model in",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )

    def test_initialise_creates_a_wandb_sweep_and_optuna_study(
        self,
        tmp_path: Path,
        invoke_cli: Callable[[list[str]], Result],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sweep_yaml = tmp_path / "search_space.yaml"
        sweep_yaml.write_text(
            yaml.safe_dump(
                {
                    "name": "example",
                    "n_trials": 3,
                    "sampler": "random",
                    "parameters": {
                        "train.optimizer.lr": {
                            "type": "float",
                            "low": 1.0e-5,
                            "high": 1.0e-2,
                        }
                    },
                }
            )
        )

        def fake_sweep(_sweep_config: dict, entity: str, project: str) -> str:
            assert entity == "turing-seaice"
            assert project == "train"
            return "fake-sweep-id"

        monkeypatch.setattr(wandb, "sweep", fake_sweep)

        result = invoke_cli(
            [
                "sweep",
                "initialise",
                "--config-name",
                "sample",
                "--sweep-yaml",
                str(sweep_yaml),
                f"base_path={tmp_path}",
            ]
        )

        assert result.exit_code == 0, result.output
        study_path = tmp_path / "sweeps" / "fake-sweep-id"
        assert (study_path / "model_config.yaml").exists()
        saved_sweep_cfg = yaml.safe_load((study_path / "optuna.yaml").read_text())
        assert saved_sweep_cfg["entity"] == "turing-seaice"
        assert saved_sweep_cfg["name"] == "example"
        assert saved_sweep_cfg["n_trials"] == 3

    def test_initialise_rejects_an_unresolvable_parameter(
        self,
        tmp_path: Path,
        invoke_cli: Callable[[list[str]], Result],
    ) -> None:
        sweep_yaml = tmp_path / "search_space.yaml"
        sweep_yaml.write_text(
            yaml.safe_dump(
                {
                    "name": "example",
                    "n_trials": 3,
                    "sampler": "random",
                    "parameters": {
                        "train.optimizer.does_not_exist": {
                            "type": "float",
                            "low": 1.0e-5,
                            "high": 1.0e-2,
                        }
                    },
                }
            )
        )

        result = invoke_cli(
            [
                "sweep",
                "initialise",
                "--config-name",
                "sample",
                "--sweep-yaml",
                str(sweep_yaml),
                f"base_path={tmp_path}",
            ]
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert not (tmp_path / "sweeps").exists()

    def test_missing_sweep_path_raises(
        self, tmp_path: Path, invoke_cli: Callable[[list[str]], Result]
    ) -> None:
        result = invoke_cli(
            ["sweep", "summarise", "--sweep-path", str(tmp_path / "missing")]
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    def test_reports_best_trial_number_and_value(
        self,
        tmp_path: Path,
        invoke_cli: Callable[[list[str]], Result],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        study_path, trial_number = self._build_study(tmp_path)
        with caplog.at_level(logging.INFO):
            result = invoke_cli(["sweep", "summarise", "--sweep-path", str(study_path)])
        assert result.exit_code == 0, result.output
        assert (
            f"Trial {trial_number} performed best, with loss 0.500000." in caplog.text
        )

    def test_reports_best_trial_parameters(
        self,
        tmp_path: Path,
        invoke_cli: Callable[[list[str]], Result],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        study_path, _ = self._build_study(tmp_path)
        with caplog.at_level(logging.INFO):
            result = invoke_cli(["sweep", "summarise", "--sweep-path", str(study_path)])
        assert result.exit_code == 0, result.output
        assert "Best trial parameters:" in caplog.text
        assert "train.optimizer.lr" in caplog.text

    def test_reports_trial_count(
        self,
        tmp_path: Path,
        invoke_cli: Callable[[list[str]], Result],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        study_path, _ = self._build_study(tmp_path)
        with caplog.at_level(logging.INFO):
            result = invoke_cli(["sweep", "summarise", "--sweep-path", str(study_path)])
        assert result.exit_code == 0, result.output
        assert "Study contains 1 trial(s)" in caplog.text

    def test_reports_no_trials_completed_without_crashing(
        self,
        tmp_path: Path,
        invoke_cli: Callable[[list[str]], Result],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        study_path, _ = self._build_study(tmp_path, n_completed=0)
        with caplog.at_level(logging.INFO):
            result = invoke_cli(["sweep", "summarise", "--sweep-path", str(study_path)])
        assert result.exit_code == 0, result.output
        assert "Study contains 0 trial(s)" in caplog.text
        assert "No trials have completed yet" in caplog.text

    def test_trial_marks_study_failed_instead_of_leaving_it_running_on_a_crash(
        self,
        tmp_path: Path,
        invoke_cli: Callable[[list[str]], Result],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A crash while training must not leave the trial RUNNING forever."""
        study_path, _ = self._build_study(tmp_path, n_completed=0)

        def _raise_from_config(_config: object) -> ModelService:
            msg = "Simulated training crash."
            raise RuntimeError(msg)

        monkeypatch.setattr(ModelService, "from_config", _raise_from_config)

        result = invoke_cli(["sweep", "trial", "--sweep-path", str(study_path)])

        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)

        trials = OptunaSweep.from_path(study_path).study.get_trials()
        assert len(trials) == 1
        assert trials[0].state == TrialState.FAIL

    def test_trial_records_the_best_checkpoint_score(
        self,
        tmp_path: Path,
        invoke_cli: Callable[[list[str]], Result],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        study_path, _ = self._build_study(tmp_path, n_completed=0)

        checkpoint = MagicMock(spec=ModelCheckpoint)
        checkpoint.best_model_score = torch.tensor(0.42)
        trainer = FakeTrainer(checkpoint_callbacks=[checkpoint])

        monkeypatch.setattr(
            ModelService, "from_config", lambda _config: FakeModelService(trainer)
        )

        with caplog.at_level(logging.INFO):
            result = invoke_cli(["sweep", "trial", "--sweep-path", str(study_path)])

        assert result.exit_code == 0, result.output
        assert "Trial 0 completed with value 0.420000" in caplog.text
        assert "Best trial (0) completed with value 0.420000." in caplog.text

        trials = OptunaSweep.from_path(study_path).study.get_trials()
        assert len(trials) == 1
        assert trials[0].state == TrialState.COMPLETE
        assert trials[0].value == pytest.approx(0.42)

    def test_trial_marks_failed_when_no_unique_checkpoint_callback(
        self,
        tmp_path: Path,
        invoke_cli: Callable[[list[str]], Result],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        study_path, _ = self._build_study(tmp_path, n_completed=0)
        trainer = FakeTrainer(checkpoint_callbacks=[])

        monkeypatch.setattr(
            ModelService, "from_config", lambda _config: FakeModelService(trainer)
        )

        with caplog.at_level(logging.WARNING):
            result = invoke_cli(["sweep", "trial", "--sweep-path", str(study_path)])

        assert result.exit_code == 0, result.output
        assert "could not find a unique ModelCheckpoint callback" in caplog.text

        trials = OptunaSweep.from_path(study_path).study.get_trials()
        assert len(trials) == 1
        assert trials[0].state == TrialState.FAIL

    def test_trial_marks_failed_when_checkpoint_has_no_best_score(
        self,
        tmp_path: Path,
        invoke_cli: Callable[[list[str]], Result],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        study_path, _ = self._build_study(tmp_path, n_completed=0)

        checkpoint = MagicMock(spec=ModelCheckpoint)
        checkpoint.best_model_score = None
        checkpoint.monitor = "validation_loss"
        trainer = FakeTrainer(checkpoint_callbacks=[checkpoint])

        monkeypatch.setattr(
            ModelService, "from_config", lambda _config: FakeModelService(trainer)
        )

        with caplog.at_level(logging.WARNING):
            result = invoke_cli(["sweep", "trial", "--sweep-path", str(study_path)])

        assert result.exit_code == 0, result.output
        assert "has no best_model_score" in caplog.text

        trials = OptunaSweep.from_path(study_path).study.get_trials()
        assert len(trials) == 1
        assert trials[0].state == TrialState.FAIL
