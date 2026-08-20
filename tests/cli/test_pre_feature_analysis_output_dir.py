"""Tests for pre-feature-analysis output-directory auto-namespacing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from icenet_mp.cli.main import app

runner = CliRunner()


class TestOutputDirAutoNamespacing:
    """The --output-dir default should be namespaced by --config-name."""

    def test_default_output_dir_is_namespaced_by_config_name(self) -> None:
        """With no --output-dir, the config name becomes the output subfolder."""
        with patch(
            "icenet_mp.cli.pre_feature_analysis._run_all_strands"
        ) as mock_run_all_strands:
            result = runner.invoke(
                app, ["pre-feature-analysis", "--config-name", "sample"]
            )

        assert result.exit_code == 0, result.output
        mock_run_all_strands.assert_called_once()
        output_dir = mock_run_all_strands.call_args.args[1]
        assert output_dir == Path("outputs/pre_feature_analysis/sample")

    def test_explicit_output_dir_overrides_the_default(self) -> None:
        """An explicit --output-dir is used verbatim, ignoring --config-name."""
        with patch(
            "icenet_mp.cli.pre_feature_analysis._run_all_strands"
        ) as mock_run_all_strands:
            result = runner.invoke(
                app,
                [
                    "pre-feature-analysis",
                    "--config-name",
                    "sample",
                    "--output-dir",
                    "outputs/custom_run",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_run_all_strands.assert_called_once()
        output_dir = mock_run_all_strands.call_args.args[1]
        assert output_dir == Path("outputs/custom_run")
