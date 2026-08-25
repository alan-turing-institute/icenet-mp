import re
from collections.abc import Callable, Sequence

import pytest
from typer.testing import CliRunner, Result

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


@pytest.fixture
def runner() -> CustomCliRunner:
    """A custom CLI runner for IceNet-MP tests."""
    return CustomCliRunner()


@pytest.fixture
def invoke_cli() -> Callable[[Sequence[str]], Result]:
    """Invoke the imp CLI app with the given arguments; asserts nothing about the result."""

    def _invoke(args: Sequence[str]) -> Result:
        return CliRunner().invoke(app, args, prog_name="imp")

    return _invoke
