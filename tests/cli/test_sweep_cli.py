import logging
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from optuna.trial import TrialState
from typer.testing import CliRunner

from icenet_mp.cli.main import app
from icenet_mp.model_service import ModelService
from icenet_mp.sweep import OptunaSweep

from .conftest import CustomCliRunner


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

    def test_missing_sweep_path_raises(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["sweep", "summarise", "--sweep-path", str(tmp_path / "missing")],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    def test_reports_best_trial_number_and_value(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        study_path, trial_number = self._build_study(tmp_path)
        runner = CustomCliRunner()
        with caplog.at_level(logging.INFO):
            runner.output(["sweep", "summarise", "--sweep-path", str(study_path)])
        assert (
            f"Trial {trial_number} performed best, with loss 0.500000." in caplog.text
        )

    def test_reports_best_trial_parameters(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        study_path, _ = self._build_study(tmp_path)
        runner = CustomCliRunner()
        with caplog.at_level(logging.INFO):
            runner.output(["sweep", "summarise", "--sweep-path", str(study_path)])
        assert "Best trial parameters:" in caplog.text
        assert "train.optimizer.lr" in caplog.text

    def test_reports_trial_count(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        study_path, _ = self._build_study(tmp_path)
        runner = CustomCliRunner()
        with caplog.at_level(logging.INFO):
            runner.output(["sweep", "summarise", "--sweep-path", str(study_path)])
        assert "Study contains 1 trial(s)" in caplog.text

    def test_reports_no_trials_completed_without_crashing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        study_path, _ = self._build_study(tmp_path, n_completed=0)
        runner = CustomCliRunner()
        with caplog.at_level(logging.INFO):
            runner.output(["sweep", "summarise", "--sweep-path", str(study_path)])
        assert "Study contains 0 trial(s)" in caplog.text
        assert "No trials have completed yet" in caplog.text

    def test_trial_marks_study_failed_instead_of_leaving_it_running_on_a_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash while training must not leave the trial RUNNING forever."""
        study_path, _ = self._build_study(tmp_path, n_completed=0)

        def _raise_from_config(_config: object) -> ModelService:
            msg = "Simulated training crash."
            raise RuntimeError(msg)

        monkeypatch.setattr(ModelService, "from_config", _raise_from_config)

        runner = CliRunner()
        result = runner.invoke(app, ["sweep", "trial", "--sweep-path", str(study_path)])

        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)

        trials = OptunaSweep.from_path(study_path).study.get_trials()
        assert len(trials) == 1
        assert trials[0].state == TrialState.FAIL
