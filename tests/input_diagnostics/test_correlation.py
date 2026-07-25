"""Tests for icenet_mp.input_diagnostics.correlation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from icenet_mp.input_diagnostics.correlation import (
    compute_correlation_matrix,
    plot_correlation_heatmap,
    print_correlation_summary,
    save_correlation_csv,
)


class TestComputeCorrelationMatrix:
    """Tests for the compute_correlation_matrix function."""

    def test_identity_correlation(self) -> None:
        """A variable should correlate perfectly with itself (1.0)."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]

        corr_df = compute_correlation_matrix(matrix, names)

        assert abs(float(corr_df.loc["a", "a"]) - 1.0) < 1e-6  # type: ignore[arg-type]
        assert abs(float(corr_df.loc["b", "b"]) - 1.0) < 1e-6  # type: ignore[arg-type]
        assert abs(float(corr_df.loc["c", "c"]) - 1.0) < 1e-6  # type: ignore[arg-type]

    def test_symmetric(self) -> None:
        """Correlation matrix should be symmetric."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]

        corr_df = compute_correlation_matrix(matrix, names)

        assert abs(float(corr_df.loc["a", "b"]) - float(corr_df.loc["b", "a"])) < 1e-6  # type: ignore[arg-type]

    def test_perfectly_correlated(self) -> None:
        """Two identical columns should have correlation of 1.0."""
        rng = np.random.default_rng(42)
        col = rng.standard_normal(100)
        matrix = np.column_stack([col, col, rng.standard_normal(100)])
        names = ["x", "y", "z"]

        corr_df = compute_correlation_matrix(matrix, names)

        assert abs(float(corr_df.loc["x", "y"]) - 1.0) < 1e-6  # type: ignore[arg-type]

    def test_values_in_range(self) -> None:
        """All correlations should be in [-1, 1]."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((500, 8))
        names = [f"v{i}" for i in range(8)]

        corr_df = compute_correlation_matrix(matrix, names)

        assert (corr_df.values >= -1.0).all()
        assert (corr_df.values <= 1.0).all()


class TestPlotCorrelationHeatmap:
    """Tests for the plot_correlation_heatmap function."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """Should create a PNG file at the given path."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]

        corr_df = compute_correlation_matrix(matrix, names)

        out_path = tmp_path / "correlations.png"
        plot_correlation_heatmap(corr_df, out_path)

        assert out_path.exists()


class TestPrintCorrelationSummary:
    """Tests for the print_correlation_summary function."""

    def test_prints_top_correlations(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Output should contain top correlation pairs."""
        rng = np.random.default_rng(42)
        col1 = rng.standard_normal(100)
        matrix = np.column_stack([col1, col1 * 0.95 + rng.standard_normal(100) * 0.3, rng.standard_normal(100)])
        names = ["x", "y", "z"]

        corr_df = compute_correlation_matrix(matrix, names)

        print_correlation_summary(corr_df, top_n=2)
        captured = capsys.readouterr()

        assert "Top 2 Correlations:" in captured.out


class TestSaveCorrelationCSV:
    """Tests for the save_correlation_csv function."""

    def test_creates_csv(self, tmp_path: Path) -> None:
        """Should create a CSV file at the given path."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]

        corr_df = compute_correlation_matrix(matrix, names)

        out_path = save_correlation_csv(corr_df, tmp_path / "corr")

        assert out_path.exists()
        assert str(out_path).endswith("correlations.csv")

    def test_csv_contains_header(self, tmp_path: Path) -> None:
        """CSV should contain variable names as header."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]

        corr_df = compute_correlation_matrix(matrix, names)

        save_correlation_csv(corr_df, tmp_path / "corr")

        csv_text = (tmp_path / "corr" / "correlations.csv").read_text()
        assert "a" in csv_text
        assert "b" in csv_text
        assert "c" in csv_text
