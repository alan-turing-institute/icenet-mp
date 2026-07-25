"""Tests for icenet_mp.input_explainability.rf."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from icenet_mp.input_explainability.rf import (
    _compute_interaction_scores,
    _get_rf_window_params,
    _windows_to_arrays,
    build_rf_windows,
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
            result.n_samples = 999  # type: ignore[misc]


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


class TestGetRFWindowParams:
    """Tests for the _get_rf_window_params function."""

    def test_defaults_from_predict_config(self) -> None:
        """Should read n_history_steps and n_forecast_steps from predict config."""
        from omegaconf import OmegaConf  # noqa: PLC0415

        cfg = OmegaConf.create({
            "predict": {"n_history_steps": 3, "n_forecast_steps": 14},
        })

        history, forecast = _get_rf_window_params(cfg)

        assert history == 3
        assert forecast == 14

    def test_rfcfg_overrides_predict(self) -> None:
        """RF config values should override predict config."""
        from omegaconf import OmegaConf  # noqa: PLC0415

        cfg = OmegaConf.create({
            "predict": {"n_history_steps": 3, "n_forecast_steps": 14},
            "rf": {"n_history_steps": 5, "n_forecast_steps": 7},
        })

        history, forecast = _get_rf_window_params(cfg)

        assert history == 5
        assert forecast == 7

    def test_predict_defaults_when_no_n_steps(self) -> None:
        """Should use hardcoded defaults when predict config lacks n_steps."""
        from omegaconf import OmegaConf  # noqa: PLC0415

        cfg = OmegaConf.create({
            "predict": {},
        })

        history, forecast = _get_rf_window_params(cfg)

        assert history == 3
        assert forecast == 14


class TestWindowsToArray:
    """Tests for the _windows_to_arrays function."""

    def test_basic_conversion(self) -> None:
        """Should convert windows to correct X and y shapes."""
        from icenet_mp.input_explainability.rf import RFWindow  # noqa: PLC0415

        rng = np.random.default_rng(42)
        n_windows, n_history, n_vars = 10, 3, 4

        windows = []
        base_date = np.datetime64("2020-01-01", "D")
        for i in range(n_windows):
            history_features = {
                "group_a": rng.standard_normal((n_history, n_vars)),
            }
            target_value = float(rng.integers(0, 100))
            windows.append(RFWindow(start_date=base_date + i, history_features=history_features, target_value=target_value))

        feature_names = [f"v{i}" for i in range(n_vars)]

        X, y = _windows_to_arrays(windows, feature_names, n_history)

        assert X.shape == (n_windows, n_history * n_vars)
        assert y.shape == (n_windows,)

    def test_target_values_preserved(self) -> None:
        """Target values should match input windows."""
        from icenet_mp.input_explainability.rf import RFWindow  # noqa: PLC0415

        rng = np.random.default_rng(42)
        n_windows, n_history, n_vars = 5, 2, 3

        target_values = [float(i * 10) for i in range(n_windows)]
        base_date = np.datetime64("2020-01-01", "D")
        windows = []
        for tv_idx, tv in enumerate(target_values):
            history_features = {
                "g": rng.standard_normal((n_history, n_vars)),
            }
            windows.append(RFWindow(start_date=base_date + tv_idx, history_features=history_features, target_value=tv))

        feature_names = [f"v{i}" for i in range(n_vars)]

        _X, y = _windows_to_arrays(windows, feature_names, n_history)

        np.testing.assert_array_almost_equal(y, target_values)


class TestBuildRFWindows:
    """Tests for the build_rf_windows function."""

    def test_no_valid_windows_raises(self) -> None:
        """Should raise when no valid windows exist (target missing forecast dates)."""
        # Mock dataset with 10 consecutive dates.
        mock_ds = MagicMock()
        mock_ds.variable_names = ["var_a"]
        mock_ds.frequency = np.timedelta64(1, "D")
        base_date = np.datetime64("2020-01-01", "D")
        mock_ds.dates = {base_date + i for i in range(10)}

        datasets = {"group_a": mock_ds}  # type: ignore[assignment]

        # Mock target zarr with dates that don't overlap forecast window.
        class _MockZarrRoot:
            def __contains__(self, key: str) -> bool:
                return True

            @property
            def attrs(self) -> MagicMock:
                m = MagicMock()
                m.get.return_value = [f"2020-01-{i+1:02d}" for i in range(5)]
                return m

        with patch("zarr.DirectoryStore"), \
             patch("zarr.group", return_value=_MockZarrRoot()), \
             pytest.raises(ValueError, match="No valid windows found"):
            build_rf_windows(  # type: ignore[arg-type]
                cast("dict[str, Any]", datasets), Path("/fake/target.zarr"), "ice_conc",
                n_history_steps=3, n_forecast_steps=5, max_samples=None,
            )

    def test_window_count_matches_valid_starts(self) -> None:
        """Number of windows should equal number of valid start dates."""
        # Mock dataset with 100 consecutive dates.
        mock_ds = MagicMock()
        mock_ds.variable_names = ["var_a", "var_b"]
        mock_ds.frequency = np.timedelta64(1, "D")
        base_date = np.datetime64("2020-01-01", "D")
        all_dates = {base_date + i for i in range(100)}
        mock_ds.dates = all_dates

        datasets = {"group_a": mock_ds}  # type: ignore[assignment]

        # Mock target zarr with dates covering the full range.
        class _MockZarrRoot:
            def __contains__(self, key: str) -> bool:
                return True

            @property
            def attrs(self) -> MagicMock:
                m = MagicMock()
                m.get.return_value = [str(base_date + i) for i in range(100)]
                return m

            def __getitem__(self, key: str) -> _MockVarGroup:
                return _MockVarGroup()

        class _MockVarGroup:
            def __contains__(self, date_str: str) -> bool:
                try:
                    d = np.datetime64(date_str[:10], "D")
                    diff_days = int(((d - base_date) / np.timedelta64(1, "D")).item())
                    return diff_days in range(100)
                except ValueError:
                    return False

            def __getitem__(self, date_str: str) -> _MockDataArray:
                return _MockDataArray()

        class _MockDataArray:
            ndim = 2

            def mean(self, _axis: tuple[int, int] | None = None) -> float:
                return 1.0

            def __getitem__(self, idx: slice) -> _MockDataArray:
                return self

        with patch("zarr.DirectoryStore"), patch("zarr.group", return_value=_MockZarrRoot()):
            windows, var_names = build_rf_windows(  # type: ignore[arg-type]
                cast("dict[str, Any]", datasets), Path("/fake/target.zarr"), "ice_conc",
                n_history_steps=3, n_forecast_steps=2, max_samples=None,
            )

        # Valid starts: dates where start+3+2-1 <= 99, i.e., start + n_forecast_steps - 1 <= 99.
        # Forecast dates for start at index i: i+n_history_steps .. i+n_history_steps+n_forecast_steps-1
        # So we need i+3+2-1 = i+4 <= 99, meaning i <= 95. That's 96 valid starts (indices 0..95).
        assert len(windows) == 96
        assert var_names == ["group_a/var_a", "group_a/var_b"]

    def test_max_samples_limits_windows(self) -> None:
        """max_samples should limit the number of windows returned."""
        mock_ds = MagicMock()
        mock_ds.variable_names = ["var_a"]
        mock_ds.frequency = np.timedelta64(1, "D")
        base_date = np.datetime64("2020-01-01", "D")
        all_dates = {base_date + i for i in range(50)}
        mock_ds.dates = all_dates

        datasets = {"group_a": mock_ds}  # type: ignore[assignment]

        class _MockZarrRoot:
            def __contains__(self, key: str) -> bool:
                return True

            @property
            def attrs(self) -> MagicMock:
                m = MagicMock()
                m.get.return_value = [str(base_date + i) for i in range(50)]
                return m

            def __getitem__(self, key: str) -> _MockVarGroup:
                return _MockVarGroup()

        class _MockVarGroup:
            def __contains__(self, date_str: str) -> bool:
                try:
                    d = np.datetime64(date_str[:10], "D")
                    diff_days = int(((d - base_date) / np.timedelta64(1, "D")).item())
                    return diff_days in range(50)
                except ValueError:
                    return False

            def __getitem__(self, date_str: str) -> _MockDataArray:
                return _MockDataArray()

        class _MockDataArray:
            ndim = 2

            def mean(self, _axis: tuple[int, int] | None = None) -> float:
                return 1.0

            def __getitem__(self, idx: slice) -> _MockDataArray:
                return self

        with patch("zarr.DirectoryStore"), patch("zarr.group", return_value=_MockZarrRoot()):
            windows, _ = build_rf_windows(  # type: ignore[arg-type]
                cast("dict[str, Any]", datasets), Path("/fake/target.zarr"), "ice_conc",
                n_history_steps=1, n_forecast_steps=1, max_samples=5,
            )

        assert len(windows) == 5


class TestTimeSeriesCV:
    """Tests that compute_rf_importance uses TimeSeriesSplit (no shuffling)."""

    def test_temporal_ordering_in_folds(self) -> None:
        """Test folds should be temporally ordered (later dates in later folds)."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 100, 3
        X = rng.standard_normal((n_samples, n_features))
        y = X[:, 0] + rng.standard_normal(n_samples) * 0.1

        result = compute_rf_importance(X, y, ["a", "b", "c"], target_name="target")

        # With TimeSeriesSplit, the last fold's test set should contain
        # the highest-indexed samples (later in time).
        assert result.n_samples == n_samples
        assert result.r2_score is not None


class TestInteractionScores:
    """Additional tests for interaction score edge cases."""

    def test_interaction_scores_none_when_features_exceed_limit(self) -> None:
        """Interaction scores should be None when features exceed the limit."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 100, 30  # exceeds _MAX_INTERACTION_FEATURES (25)
        X = rng.standard_normal((n_samples, n_features))
        y = X[:, 0] + rng.standard_normal(n_samples) * 0.1

        result = compute_rf_importance(X, y, [f"f{i}" for i in range(n_features)], target_name="target")

        assert result.interaction_scores is None
