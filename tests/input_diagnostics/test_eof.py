"""Tests for icenet_mp.input_diagnostics.eof."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from icenet_mp.input_diagnostics.eof import (
    compute_eof,
    print_eof_table,
    save_eof_results,
)


class TestComputeEOF:
    """Tests for the compute_eof function."""

    def test_independent_variables_equal_importance(self) -> None:
        """Independent variables should have roughly equal importance scores."""
        rng = np.random.default_rng(42)
        n_samples, n_vars = 500, 5
        matrix = rng.standard_normal((n_samples, n_vars))
        names = [f"var_{i}" for i in range(n_vars)]

        result = compute_eof(matrix, names)

        np.testing.assert_allclose(
            result.feature_importance,
            result.feature_importance.mean(),
            atol=0.15,
        )
        assert result.n_samples == n_samples
        assert result.n_features == n_vars

    def test_one_dominant_variable(self) -> None:
        """A variable that dominates variance should have highest importance."""
        rng = np.random.default_rng(42)
        n_samples = 500
        dominant = rng.standard_normal(n_samples) * 100
        others = rng.standard_normal((n_samples, 4))

        matrix = np.column_stack([dominant, others])
        names = ["dominant", "a", "b", "c", "d"]

        result = compute_eof(matrix, names)

        assert result.feature_importance[0] > result.feature_importance[1:].max() * 2

    def test_perfectly_correlated_variables(self) -> None:
        """Two perfectly correlated variables should share high importance."""
        rng = np.random.default_rng(42)
        n_samples = 300
        col = rng.standard_normal(n_samples)
        matrix = np.column_stack([col, col, rng.standard_normal(n_samples)])
        names = ["x", "y", "z"]

        result = compute_eof(matrix, names)

        # Correlated pair should have similar importance.
        np.testing.assert_allclose(
            result.feature_importance[0], result.feature_importance[1], rtol=0.05
        )
        assert result.feature_importance[0] > result.feature_importance[2]

    def test_explained_variance_sums_to_one(self) -> None:
        """Explained variance ratios should sum to ~1.0."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((200, 5))
        names = [f"v{i}" for i in range(5)]

        result = compute_eof(matrix, names)

        np.testing.assert_allclose(
            result.explained_variance_ratio.sum(), 1.0, atol=1e-6
        )
        assert abs(result.cumulative_explained_variance[-1] - 1.0) < 1e-6

    def test_eof_modes_shape(self) -> None:
        """EOF modes shape should be (n_modes, n_features)."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((300, 8))
        names = [f"v{i}" for i in range(8)]

        result = compute_eof(matrix, names)

        assert result.eof_modes.shape == (min(300, 8), 8)

    def test_feature_names_preserved(self) -> None:
        """Variable names should match input."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["alpha", "beta", "gamma"]

        result = compute_eof(matrix, names)

        assert result.variable_names == names


class TestComputeEOFErrors:
    """Tests for error conditions in compute_eof."""

    def test_single_variable_raises(self) -> None:
        """EOF should raise ValueError with only 1 variable."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 1))

        with pytest.raises(ValueError, match="at least 2 timesteps and 2 variables"):
            compute_eof(matrix, ["x"])

    def test_single_timestep_raises(self) -> None:
        """EOF should raise ValueError with only 1 timestep."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((1, 5))

        with pytest.raises(ValueError, match="at least 2 timesteps and 2 variables"):
            compute_eof(matrix, ["a", "b", "c", "d", "e"])

    def test_empty_matrix_raises(self) -> None:
        """EOF should raise ValueError on empty matrix."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((0, 3))

        with pytest.raises(ValueError, match="at least 2 timesteps"):
            compute_eof(matrix, ["a", "b", "c"])


class TestEOFResult:
    """Tests for the EOFResult dataclass."""

    def test_frozen(self) -> None:
        """EOFResult should be immutable."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["x", "y", "z"]
        result = compute_eof(matrix, names)

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.n_samples = 999  # type: ignore[misc]


class TestSaveEOFResults:
    """Tests for the save_eof_results function."""

    def test_creates_json_and_txt(self, tmp_path: Path) -> None:
        """save_eof_results should create JSON and TXT files."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]
        result = compute_eof(matrix, names)

        json_path = save_eof_results(result, tmp_path / "eof")

        assert json_path.exists()
        assert (tmp_path / "eof" / "eof_report.txt").exists()

    def test_json_is_valid(self, tmp_path: Path) -> None:
        """JSON file should be valid and contain expected keys."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]
        result = compute_eof(matrix, names)

        save_eof_results(result, tmp_path / "eof")

        with (tmp_path / "eof" / "eof_results.json").open() as fh:
            data = json.load(fh)

        assert "variable_names" in data
        assert "explained_variance_ratio" in data
        assert "eof_modes" in data
        assert "feature_importance" in data
        assert data["n_samples"] == 100
        assert (
            data["analysis_space"]
            == "variable-space covariance decomposition over spatial means"
        )
        assert data["standardised"] is False
        assert data["independent_feature_selection_evidence"] is False


class TestPrintEOFTable:
    """Tests for the print_eof_table function."""

    def test_prints_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Output should contain expected sections."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]
        result = compute_eof(matrix, names)

        print_eof_table(result)
        captured = capsys.readouterr()

        assert "EOF Variable-Space Covariance Decomposition" in captured.out
        assert "not a spatial EOF" in captured.out
        assert "Explained Variance:" in captured.out
        assert "Feature Importance" in captured.out
        assert "a" in captured.out
        assert "b" in captured.out
        assert "c" in captured.out
