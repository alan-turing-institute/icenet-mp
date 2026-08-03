"""Individual strand tests for icenet_mp.cli.pre_feature_analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from icenet_mp.cli.pre_feature_analysis import (
    _run_correlation_strand,
    _run_eof_strand,
    _run_pca_strand,
    _run_vif_strand,
)


class TestVIFStrand:
    """Tests for the VIF strand."""

    def test_multi_variable_succeeds(self, tmp_path: Path) -> None:
        """Multi-variable input should produce valid VIF results."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 5))
        names = [f"var_{i}" for i in range(5)]

        _run_vif_strand(matrix, names, threshold=5.0, output_dir=tmp_path)

        assert (tmp_path / "vif" / "vif_results.json").exists()
        assert (tmp_path / "vif" / "vif_report.txt").exists()

    def test_single_variable_skips_gracefully(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Single-variable input should skip VIF with a warning."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 1))

        _run_vif_strand(
            matrix, ["only_var"], threshold=5.0, output_dir=Path("/tmp/test")
        )

        captured = capsys.readouterr()
        assert (
            "skipped" in captured.err.lower() or "at least 2 variables" in captured.err
        )


class TestPCAStrand:
    """Tests for the PCA strand."""

    def test_multi_variable_succeeds(self, tmp_path: Path) -> None:
        """Multi-variable input should produce valid PCA results."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 5))
        names = [f"var_{i}" for i in range(5)]

        _run_pca_strand(matrix, names, output_dir=tmp_path)

        assert (tmp_path / "pca" / "pca_results.json").exists()
        assert (tmp_path / "pca" / "pca_report.txt").exists()

    def test_single_variable_skips_gracefully(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Single-variable input should skip PCA with a warning."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 1))

        _run_pca_strand(matrix, ["only_var"], output_dir=Path("/tmp/test"))  # noqa: S108

        captured = capsys.readouterr()
        assert (
            "skipped" in captured.err.lower() or "at least 2 variables" in captured.err
        )


class TestEOFStrand:
    """Tests for the EOF strand."""

    def test_multi_variable_succeeds(self, tmp_path: Path) -> None:
        """Multi-variable input should produce valid EOF results."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 5))
        names = [f"var_{i}" for i in range(5)]

        _run_eof_strand(matrix, names, output_dir=tmp_path)

        assert (tmp_path / "eof" / "eof_results.json").exists()
        assert (tmp_path / "eof" / "eof_report.txt").exists()

    def test_single_variable_raises(self) -> None:
        """EOF strand should propagate ValueError for single variable."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 1))

        with pytest.raises(ValueError, match="at least 2 timesteps and 2 variables"):
            _run_eof_strand(matrix, ["only_var"], output_dir=Path("/tmp/test"))  # noqa: S108


class TestCorrelationStrand:
    """Tests for the correlation strand."""

    def test_multi_variable_succeeds(self, tmp_path: Path) -> None:
        """Multi-variable input should produce correlation outputs.

        Note: _run_correlation_strand passes output_dir / "correlations" to both
        plot and save functions, creating a nested structure:
        <output_dir>/correlations/correlations.csv (CSV) + correlations.png (PNG).
        """
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 5))
        names = [f"var_{i}" for i in range(5)]

        _run_correlation_strand(matrix, names, output_dir=tmp_path)

        nested_corr = tmp_path / "correlations"
        assert (nested_corr / "correlations.csv").exists()
        # plot_correlation_heatmap saves without .png extension; check directory has files.
        assert any(p.is_file() for p in nested_corr.iterdir())

    def test_single_variable_succeeds(self, tmp_path: Path) -> None:
        """Single-variable input should still produce correlation output."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 1))

        _run_correlation_strand(matrix, ["only_var"], output_dir=tmp_path)

        nested_corr = tmp_path / "correlations"
        assert (nested_corr / "correlations.csv").exists()


class TestStrandLoggingOutput:
    """Tests for logging and console output from strands."""

    def test_vif_prints_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        """VIF strand should print a results table to stdout."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]

        _run_vif_strand(matrix, names, threshold=5.0, output_dir=Path("/tmp/test"))  # noqa: S108

        captured = capsys.readouterr()
        assert "VIF" in captured.out or "VIF" in captured.err
        assert any(name in captured.out for name in names)

    def test_pca_prints_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        """PCA strand should print a results table to stdout."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]

        _run_pca_strand(matrix, names, output_dir=Path("/tmp/test"))  # noqa: S108

        captured = capsys.readouterr()
        assert "PCA" in captured.out or "PCA" in captured.err
        assert any(name in captured.out for name in names)

    def test_eof_prints_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        """EOF strand should print a results table to stdout."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]

        _run_eof_strand(matrix, names, output_dir=Path("/tmp/test"))  # noqa: S108

        captured = capsys.readouterr()
        assert "EOF" in captured.out or "EOF" in captured.err
        assert any(name in captured.out for name in names)
