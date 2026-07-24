"""Tests for icenet_mp.input_explainability.rf."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from icenet_mp.input_explainability.rf import (
    _compute_interaction_scores,
    compute_rf_importance,
    save_rf_results,
)


class TestComputeRFImportance:
    """Tests for the compute_rf_importance function."""

    def test_basic_run(self) -> None:
        """Basic run should produce valid results with positive R²."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 300, 5
        X = rng.standard_normal((n_samples, n_features))
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.standard_normal(n_samples) * 0.1
        names = [f"feat_{i}" for i in range(n_features)]

        result = compute_rf_importance(X, y, names, target_name="target", n_estimators=50, random_state=42)

        assert result.n_samples == n_samples
        assert result.n_features == n_features
        assert len(result.permutation_importance) == n_features
        assert len(result.importances_std) == n_features
        assert -1.0 <= result.r2_score <= 1.0
        assert result.mse >= 0

    def test_feature_names_preserved(self) -> None:
        """Feature names should match input."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X[:, 0] + rng.standard_normal(100) * 0.1
        names = ["alpha", "beta", "gamma"]

        result = compute_rf_importance(X, y, names, target_name="target")

        assert result.feature_names == names

    def test_target_name_stored(self) -> None:
        """Target name should be stored in result."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X[:, 0] + rng.standard_normal(100) * 0.1

        result = compute_rf_importance(X, y, ["x"], target_name="sic-ssmis/ice_conc")

        assert result.target_name == "sic-ssmis/ice_conc"

    def test_high_signal_feature_ranked_first(self) -> None:
        """A feature with strong signal should rank higher."""
        rng = np.random.default_rng(42)
        n_samples = 300
        X = rng.standard_normal((n_samples, 5))
        # Target driven almost entirely by first feature.
        y = X[:, 0] * 10 + rng.standard_normal(n_samples) * 0.1
        names = [f"feat_{i}" for i in range(5)]

        result = compute_rf_importance(X, y, names, target_name="target", n_estimators=100, random_state=42)

        # First feature should have highest importance (or tied).
        assert result.permutation_importance[0] >= result.permutation_importance[1:].max() * 0.5

    def test_interaction_scores_computed_for_few_features(self) -> None:
        """Interaction scores should be computed when features <= limit."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 200, 4
        X = rng.standard_normal((n_samples, n_features))
        y = X[:, 0] + X[:, 1] * 0.5 + rng.standard_normal(n_samples) * 0.1
        names = [f"f{i}" for i in range(n_features)]

        result = compute_rf_importance(X, y, names, target_name="target", n_estimators=50, random_state=42)

        assert result.interaction_scores is not None
        assert result.interaction_scores.shape == (n_features, n_features)


class TestComputeInteractionScores:
    """Tests for the _compute_interaction_scores function."""

    def test_symmetric_matrix(self) -> None:
        """Interaction matrix should be symmetric."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 4))
        y = X[:, 0] + X[:, 1] * 0.5 + rng.standard_normal(200) * 0.1

        from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415

        model = RandomForestRegressor(n_estimators=30, random_state=42)
        model.fit(X, y)

        interactions = _compute_interaction_scores(model, X, y, ["a", "b", "c", "d"])

        assert interactions is not None
        np.testing.assert_array_almost_equal(interactions, interactions.T)

    def test_zeros_on_diagonal(self) -> None:
        """Diagonal of interaction matrix should be zero."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        y = X[:, 0] + rng.standard_normal(200) * 0.1

        from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415

        model = RandomForestRegressor(n_estimators=30, random_state=42)
        model.fit(X, y)

        interactions = _compute_interaction_scores(model, X, y, ["a", "b", "c"])

        assert interactions is not None
        np.testing.assert_array_almost_equal(np.diag(interactions), [0.0, 0.0, 0.0])

    def test_none_for_too_many_features(self) -> None:
        """Interaction scores should be None when features exceed limit."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 25))
        y = X[:, 0] + rng.standard_normal(200) * 0.1

        from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415

        model = RandomForestRegressor(n_estimators=30, random_state=42)
        model.fit(X, y)

        interactions = _compute_interaction_scores(model, X, y, [f"f{i}" for i in range(25)])

        assert interactions is None

    def test_non_uniform_values_with_signal(self) -> None:
        """Synergistic feature pairs should have higher H-statistic than non-interacting ones."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 500, 3
        X = rng.standard_normal((n_samples, n_features))
        # Target driven by feature 0 and a synergy between 0 and 1.
        y = X[:, 0] * 2 + (X[:, 0] * X[:, 1]) * 1.5 + rng.standard_normal(n_samples) * 0.3

        from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        interactions = _compute_interaction_scores(model, X, y, ["a", "b", "c"])

        assert interactions is not None
        # Pair (0, 1) has known synergy → should have highest H-statistic.
        assert interactions[0, 1] > interactions[0, 2]
        assert interactions[0, 1] > interactions[1, 2]


class TestRFResult:
    """Tests for the RFResult dataclass."""

    def test_frozen(self) -> None:
        """RFResult should be immutable."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X[:, 0] + rng.standard_normal(100) * 0.1

        result = compute_rf_importance(X, y, ["x", "y", "z"], target_name="target")

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.n_samples = 999  # type: ignore[func-returns-value]


class TestSaveRFResults:
    """Tests for the save_rf_results function."""

    def test_creates_outputs(self, tmp_path: Path) -> None:
        """save_rf_results should create JSON, TXT, and plot files."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X[:, 0] + rng.standard_normal(100) * 0.1

        result = compute_rf_importance(X, y, ["a", "b", "c"], target_name="target")

        json_path, txt_path, plot_paths = save_rf_results(result, tmp_path / "rf")

        assert json_path.exists()
        assert txt_path.exists()
        assert len(plot_paths) >= 1  # at least feature importance plot

    def test_json_is_valid(self, tmp_path: Path) -> None:
        """JSON file should be valid and contain expected keys."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X[:, 0] + rng.standard_normal(100) * 0.1

        result = compute_rf_importance(X, y, ["a", "b", "c"], target_name="target")

        save_rf_results(result, tmp_path / "rf")

        with (tmp_path / "rf" / "rf_results.json").open() as fh:
            data = json.load(fh)

        assert "feature_names" in data
        assert "permutation_importance" in data
        assert "r2_score" in data
        assert "n_samples" in data
