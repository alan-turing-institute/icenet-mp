from .conftest import CustomCliRunner


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

    def test_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["--help"],
            expected_patterns=self.expected_patterns_help,
        )

    def test_short_help(self, runner: CustomCliRunner) -> None:
        runner.check_output(
            ["-h"],
            expected_patterns=self.expected_patterns_help,
        )
