import re
from collections.abc import Sequence

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
            assert found_match, f"Pattern '{pattern}' not found in output"


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
                r"init\s+Create all datasets.",
                r"load\s+Load dataset in parts.",
                r"finalise\s+Finalise loaded dataset.",
            ],
        )


class TestEvaluateCLI:
    def test_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["evaluate", "--help"],
            expected_patterns=[
                r"Usage: imp evaluate \[OPTIONS\] \[OVERRIDES\]...",
                r"Evaluate a pre-trained model",
                r"overrides\s+\[OVERRIDES\]...\s+Apply space-separated Hydra config",
                r"--config-name\s+TEXT\s+Specify the name of a file to load from the",
                r"--checkpoint\s+TEXT\s+Specify the path to a trained model",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )


class TestTrainCLI:
    def test_help(self) -> None:
        runner = CustomCliRunner()
        runner.check_output(
            ["train", "--help"],
            expected_patterns=[
                r"Usage: imp train \[OPTIONS\] \[OVERRIDES\]...",
                r"Train a model",
                r"overrides\s+\[OVERRIDES\]...\s+Apply space-separated Hydra config",
                r"--config-name\s+TEXT\s+Specify the name of a file to load from the",
                r"--help\s+-h\s+Show this message and exit.",
            ],
        )
