import re
from collections.abc import Sequence
from importlib import import_module

import pytest
import typer
from typer.testing import CliRunner

from icenet_mp.cli.main import app


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
        r"train\s+Train a model",
        r"pre-feature-analysis\s+Run all input variable analysis strands",
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

    def test_mps_failure_exits_unsuccessfully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unsupported MPS operation must not be reported as a successful command."""
        cli_main = import_module("icenet_mp.cli.main")

        def raise_mps_error() -> None:
            message = "not currently implemented for the MPS device"
            raise NotImplementedError(message)

        monkeypatch.setattr(cli_main, "app", raise_mps_error)

        with pytest.raises(typer.Exit) as exc_info:
            cli_main.main()

        assert exc_info.value.exit_code == 1


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
                r"masks\s+Create land / active grid cell masks.",
            ],
        )


class TestEvaluateCLI:
    def test_evaluate_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["evaluate", "--help"],
            expected_patterns=[
                r"Usage: imp evaluate \[OPTIONS\] \[overrides\]...",
                r"Evaluate a pre-trained model",
                r"overrides\s+<str>\s+One or more space-separated Hydra config overrides",
                r"--config-name\s+<str>\s+Name of a file to load from the config",
                r"--checkpoint\s+<str>\s+Path of a trained model checkpoint",
                r"--help\s+-h\s+Show this message and exit.",
                r"--save-layer\s+<str>\s+Dotted path of a model submodule to hook",
            ],
        )


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
                r"--multistage\s+Train an EncodeProcessDecode model in",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )


class TestPreFeatureAnalysisCLI:
    def test_pre_feature_analysis_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["pre-feature-analysis", "--help"],
            expected_patterns=[
                r"Usage: imp pre-feature-analysis \[OPTIONS\] \[overrides\]...",
                r"Run all input variable analysis strands",
                r"overrides\s+<str>\s+One or more space-separated Hydra config",
                r"--config-name\s+<str>\s+Name of a file to load from the config",
                r"--output-dir\s+<str>\s+Root directory for all analysis results",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )
