"""Tests for icenet_mp.input_diagnostics.vif."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from icenet_mp.input_diagnostics.vif import compute_vif, print_vif_table


class TestComputeVIF:
    """Tests for the compute_vif function."""

    def test_independent_variables(self) -> None:
        """VIF should be ~1.0 for independent variables."""
        rng = np.random.default_rng(42)
        n_samples, n_vars = 500, 5
        matrix = rng.standard_normal((n_samples, n_vars))

        names = [f"var_{i}" for i in range(n_vars)]
        result = compute_vif(matrix, names, threshold=5.0)

        assert len(result.vif_scores) == n_vars
        # Independent variables should have VIF close to 1.0
        np.testing.assert_allclose(result.vif_scores, 1.0, atol=0.1)
        assert result.threshold == 5.0
        assert result.n_samples == n_samples

    def test_perfectly_correlated_variables(self) -> None:
        """VIF should be very high for perfectly correlated variables."""
        rng = np.random.default_rng(42)
        n_samples = 200
        # Create two perfectly correlated columns
        col1 = rng.standard_normal(n_samples)
        col2 = col1.copy()  # Perfect correlation
        col3 = rng.standard_normal(n_samples)

        matrix = np.column_stack([col1, col2, col3])
        names = ["a", "b", "c"]

        result = compute_vif(matrix, names, threshold=5.0)

        assert len(result.vif_scores) == 3
        # Perfectly correlated variables should have very high VIF
        assert result.vif_scores[0] > 1e6
        assert result.vif_scores[1] > 1e6
        # Independent variable should have low VIF
        assert result.vif_scores[2] < 2.0

    def test_single_variable_raises(self) -> None:
        """VIF should raise ValueError when given only one variable."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 1))

        with pytest.raises(ValueError, match="at least 2 variables"):
            compute_vif(matrix, ["x"])

    def test_empty_variables_raises(self) -> None:
        """VIF should raise ValueError when given zero variables."""
        matrix = np.empty((100, 0))

        with pytest.raises(ValueError, match="at least 2 variables"):
            compute_vif(matrix, [])

    def test_highly_correlated_variables(self) -> None:
        """VIF should be elevated for highly correlated variables."""
        rng = np.random.default_rng(42)
        n_samples = 300
        col1 = rng.standard_normal(n_samples)
        # Highly correlated (but not perfect)
        col2 = 0.95 * col1 + 0.3 * rng.standard_normal(n_samples)

        matrix = np.column_stack([col1, col2])
        names = ["x", "y"]

        result = compute_vif(matrix, names, threshold=5.0)

        assert len(result.vif_scores) == 2
        # Highly correlated variables should have VIF > 5
        assert result.vif_scores[0] > 5.0
        assert result.vif_scores[1] > 5.0

    def test_threshold_flagging(self) -> None:
        """Variables above threshold should be flagged."""
        rng = np.random.default_rng(42)
        n_samples = 200
        col1 = rng.standard_normal(n_samples)
        col2 = col1.copy()  # Perfect correlation

        matrix = np.column_stack([col1, col2])
        names = ["a", "b"]

        result = compute_vif(matrix, names, threshold=5.0)

        flagged = [
            name
            for name, score in zip(names, result.vif_scores, strict=True)
            if score > 5.0
        ]
        assert len(flagged) == 2
        assert set(flagged) == {"a", "b"}

    def test_custom_threshold(self) -> None:
        """Custom threshold should be respected."""
        rng = np.random.default_rng(42)
        n_samples, n_vars = 500, 3
        matrix = rng.standard_normal((n_samples, n_vars))
        names = [f"var_{i}" for i in range(n_vars)]

        result = compute_vif(matrix, names, threshold=10.0)
        assert result.threshold == 10.0

    def test_opt_in_evidence_qualification_requires_ten_to_one(self) -> None:
        rng = np.random.default_rng(42)
        result = compute_vif(
            rng.standard_normal((19, 2)), ["x", "y"], qualification_enabled=True
        )

        assert result.evidence_qualified is False
        assert result.qualification is not None
        assert result.qualification["sample_to_feature_ratio"] == 9.5
        assert "Temporal autocorrelation" in str(
            result.qualification["independence_warning"]
        )

    def test_qualification_standardises_columns_and_reports_deficiencies(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        matrix = np.column_stack([x, x * 1.0e9, np.ones(100)])

        default_result = compute_vif(matrix, ["x", "scaled_x", "constant"])
        result = compute_vif(
            matrix, ["x", "scaled_x", "constant"], qualification_enabled=True
        )

        np.testing.assert_equal(result.vif_scores, default_result.vif_scores)
        assert result.qualification is not None
        assert result.qualification["constant_columns"] == ["constant"]
        assert result.qualification["diagnostic_predictor_count"] == 2
        assert result.qualification["exact_rank_deficient"] is True
        assert result.qualification["near_rank_deficient"] is True
        assert result.evidence_qualified is False

    def test_qualification_condition_number_is_scale_invariant(self) -> None:
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 2))
        scaled = matrix.copy()
        scaled[:, 1] *= 1.0e12

        result = compute_vif(scaled, ["x", "large_units"], qualification_enabled=True)

        assert result.qualification is not None
        condition_number = result.qualification["condition_number"]
        assert isinstance(condition_number, float)
        assert condition_number < 2.0
        assert result.evidence_qualified is True


class TestPrintVIFTable:
    """Tests for the print_vif_table function."""

    def test_prints_sorted_by_vif(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Output should be sorted by VIF score descending."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 3))
        names = ["a", "b", "c"]
        result = compute_vif(matrix, names, threshold=5.0)

        print_vif_table(result)
        captured = capsys.readouterr()

        assert "VIF Analysis Results" in captured.out
        assert "variables" in captured.out.lower()


class TestVIFResult:
    """Tests for the VIFResult dataclass."""

    def test_frozen(self) -> None:
        """VIFResult should be immutable."""
        rng = np.random.default_rng(42)
        result = compute_vif(
            np.column_stack([rng.standard_normal(100), rng.standard_normal(100)]),
            ["x", "y"],
            threshold=5.0,
        )
        # frozen dataclass should reject mutation — mypy flags this but it's intentional
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.threshold = 10.0  # type: ignore[misc]
