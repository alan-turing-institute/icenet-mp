import logging
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from optuna.trial import TrialState
from typer.testing import CliRunner

from icenet_mp.cli.main import app
from icenet_mp.model_service import ModelService
from icenet_mp.sweep import OptunaSweep


class CustomCliRunner(CliRunner):
    def __init__(self) -> None:
        """A custom CLI runner for IceNet-MP tests."""
        super().__init__()
        self.colorstrip = re.compile(r"\x1b\[[0-9;]*m")

    def output(self, commands: Sequence[str]) -> list[str]:
        """Invoke the CLI commands and return the output as a list of strings."""
        result = super().invoke(app, commands, prog_name="imp")
        assert result.exit_code == 0, (
            f"Command failed with exit code {result.exit_code}: {result.output}"
        )
        if result.exception:
            raise result.exception
        return [self.colorstrip.sub("", line) for line in result.output.split("\n")]

    def check_output(
        self, commands: Sequence[str], expected_patterns: Sequence[str]
    ) -> None:
        """Check if the output contains all expected patterns."""
        output = self.output(commands)
        for pattern in expected_patterns:
            found_match = any(re.search(pattern, line) for line in output)
            assert found_match, f"Pattern '{pattern}' not found in output."


class TestBaseCLI:
    expected_patterns_help = (
        r"Usage: imp \[OPTIONS\] COMMAND \[ARGS\]...",
        r"Entrypoint for imp CLI application.",
        r"--install-completion\s+Install completion for the current shell.",
        r"--show-completion\s+Show completion for the current shell",
        r"--help\s+-h\s+Show this message and exit.",
        r"datasets\s+Manage datasets",
        r"evaluate\s+Evaluate a pre-trained model",
        r"sweep\s+Generate W&B sweeps with Optuna-sampled",
        r"train\s+Train a model",
    )

    def test_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["--help"],
            expected_patterns=self.expected_patterns_help,
        )

    def test_short_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["-h"],
            expected_patterns=self.expected_patterns_help,
        )


class TestDatasetsCLI:
    def test_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["datasets", "--help"],
            expected_patterns=[
                r"Usage: imp datasets \[OPTIONS\] COMMAND \[ARGS\]...",
                r"Manage datasets",
                r"--help\s+-h\s+Show this message and exit.",
                r"create\s+Create all datasets.",
                r"inspect\s+Inspect all datasets.",
                r"plot\s+Plot one timestep of configured datasets.",
                r"masks\s+Create land / active grid cell masks.",
            ],
        )

    def test_plot_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["datasets", "plot", "--help"],
            expected_patterns=[
                r"Usage: imp datasets plot \[OPTIONS\] \[overrides\]...",
                r"Plot one timestep of configured datasets.",
                r"--dataset\s+<str>\s+Only plot the named configured",
                r"--timestep\s+<int>\s+Dataset timestep index to plot",
                r"--video\s+--no-video\s+Animate --n-steps consecutive",
                r"--n-steps\s+<int>\s+Number of consecutive timesteps to",
            ],
        )


class TestDatasetsPlotCLI:
    class FakeDownloader:
        """A minimal downloader stub exposing only what `plot` needs."""

        def __init__(self, name: str, path_dataset: Path) -> None:
            """Store the downloader's name and dataset path."""
            self.name = name
            self.path_dataset = path_dataset

    def test_plot_calls_plot_dataset_for_each_matched_existing_dataset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Call plot_dataset once per configured downloader whose data exists."""
        existing_path = tmp_path / "example.zarr"
        existing_path.mkdir()
        downloaders = [
            self.FakeDownloader("example", existing_path),
            self.FakeDownloader("missing", tmp_path / "missing.zarr"),
        ]
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders", lambda _config: downloaders
        )

        calls = []

        def fake_plot_dataset(
            name: str, path: Path, output_dir: Path, timestep: int
        ) -> int:
            calls.append((name, path, output_dir, timestep))
            return 2

        monkeypatch.setattr(
            "icenet_mp.cli.datasets.plot_variables_static", fake_plot_dataset
        )

        result = CliRunner().invoke(
            app,
            ["datasets", "plot", f"base_path={tmp_path}", "--timestep", "1"],
            prog_name="imp",
        )

        assert result.exit_code == 0, result.output
        expected_output_dir = tmp_path.resolve() / "data" / "input_plots"
        assert calls == [("example", existing_path, expected_output_dir, 1)]

    def test_plot_video_calls_plot_dataset_video_with_n_steps(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Call plot_dataset_video instead of plot_dataset when --video is passed."""
        existing_path = tmp_path / "example.zarr"
        existing_path.mkdir()
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders",
            lambda _config: [self.FakeDownloader("example", existing_path)],
        )
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.plot_variables_static",
            lambda *_args: pytest.fail(
                "plot_variables_static should not be called with --video"
            ),
        )

        calls = []

        def fake_plot_dataset_video(
            name: str, path: Path, output_dir: Path, timestep: int, n_steps: int
        ) -> int:
            calls.append((name, path, output_dir, timestep, n_steps))
            return 4

        monkeypatch.setattr(
            "icenet_mp.cli.datasets.plot_variables_video", fake_plot_dataset_video
        )

        result = CliRunner().invoke(
            app,
            [
                "datasets",
                "plot",
                f"base_path={tmp_path}",
                "--video",
                "--n-steps",
                "5",
            ],
            prog_name="imp",
        )

        assert result.exit_code == 0, result.output
        expected_output_dir = tmp_path.resolve() / "data" / "input_plots"
        assert calls == [("example", existing_path, expected_output_dir, 0, 5)]

    def test_plot_rejects_unmatched_dataset_name(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exit non-zero when --dataset names a dataset that isn't configured."""
        monkeypatch.setattr(
            "icenet_mp.cli.datasets.build_downloaders",
            lambda _config: [self.FakeDownloader("example", tmp_path / "example.zarr")],
        )

        result = CliRunner().invoke(
            app,
            ["datasets", "plot", "--dataset", "unknown"],
            prog_name="imp",
        )

        assert result.exit_code == 1


class TestEvaluateCLI:
    def test_evaluate_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["evaluate", "--help"],
            expected_patterns=[
                r"Usage: imp evaluate \[OPTIONS\] \[overrides\]...",
                r"Evaluate a pre-trained model",
                r"overrides\s+<str>\s+One or more space-separated Hydra config overrides",
                r"--checkpoint\s+<str>\s+Path of a trained model checkpoint",
                r"--config-name\s+<str>\s+Name of a file to load from the config",
                r"--help\s+-h\s+Show this message and exit.",
                r"--save-layer\s+<str>\s+Dotted path of a model submodule to hook",
            ],
        )


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


class TestTrainCLI:
    def test_train_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["train", "--help"],
            expected_patterns=[
                r"Usage: imp train \[OPTIONS\] \[overrides\]...",
                r"Train a model",
                r"overrides\s+<str>\s+One or more space-separated Hydra config overrides",
                r"--checkpoint-dir\s+<str>\s+Path to a directory of existing",
                r"--config-name\s+<str>\s+Name of a file to load from the config",
                r"--help\s+-h\s+Show this message and exit.",
                r"--multistage\s+Train an EncodeProcessDecode model in",
            ],
        )
