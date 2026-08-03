"""Tests for icenet_mp.input_explainability.rf."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from icenet_mp.input_explainability.rf import (
    RFWindow,
    _compute_interaction_scores,
    _get_rf_window_params,
    _windows_to_arrays,
    build_rf_windows,
    compute_rf_importance,
    print_rf_table,
    save_rf_results,
)


class FakeSingleDataset:
    """Minimal in-memory dataset for RF window tests."""

    def __init__(
        self,
        name: str,
        variables: list[str],
        values: dict[np.datetime64, np.ndarray],
    ) -> None:
        """Initialise a dataset from a date-to-array mapping."""
        self.name = name
        self.variable_names = variables
        self._values = values
        self.dates = sorted(values)
        self.frequency = np.timedelta64(1, "D")

    def get_tchw(self, dates: list[np.datetime64]) -> np.ndarray:
        return np.stack([self._values[date] for date in dates])


class TestComputeRFImportance:
    """Tests for the compute_rf_importance function."""

    def test_basic_run(self) -> None:
        """Basic run should produce valid results with positive R²."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 300, 5
        X = rng.standard_normal((n_samples, n_features))
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.standard_normal(n_samples) * 0.1
        names = [f"feat_{i}" for i in range(n_features)]

        result = compute_rf_importance(
            X, y, names, target_name="target", n_estimators=50, random_state=42
        )

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
        """A feature with strong signal should rank highest."""
        rng = np.random.default_rng(42)
        n_samples = 300
        X = rng.standard_normal((n_samples, 5))
        # Target driven almost entirely by first feature.
        y = X[:, 0] * 10 + rng.standard_normal(n_samples) * 0.1
        names = [f"feat_{i}" for i in range(5)]

        result = compute_rf_importance(
            X, y, names, target_name="target", n_estimators=100, random_state=42
        )

        # First feature should have strictly highest importance.
        assert (
            result.permutation_importance[0] > result.permutation_importance[1:].max()
        )

    def test_interaction_scores_computed_for_few_features(self) -> None:
        """Interaction scores should be computed when features <= limit."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 200, 4
        X = rng.standard_normal((n_samples, n_features))
        y = X[:, 0] + X[:, 1] * 0.5 + rng.standard_normal(n_samples) * 0.1
        names = [f"f{i}" for i in range(n_features)]

        result = compute_rf_importance(
            X, y, names, target_name="target", n_estimators=50, random_state=42
        )

        assert result.interaction_scores is not None
        assert result.interaction_scores.shape == (n_features, n_features)

    def test_interaction_disabled(self) -> None:
        """When interaction_enabled=False, interaction_scores should be None."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X[:, 0] + rng.standard_normal(100) * 0.1

        result = compute_rf_importance(
            X,
            y,
            ["a", "b", "c"],
            target_name="target",
            interaction_enabled=False,
        )

        assert result.interaction_scores is None

    def test_print_rf_table_shows_interaction_section(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_rf_table should show pairwise interaction section when scores exist."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X[:, 0] + X[:, 1] * 0.5 + rng.standard_normal(100) * 0.1

        result = compute_rf_importance(X, y, ["a", "b", "c"], target_name="target")

        print_rf_table(result)
        captured = capsys.readouterr()

        assert "Pairwise Interaction Scores (Friedman H-statistic)" in captured.out

    def test_print_rf_table_no_interaction_when_disabled(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_rf_table should omit interaction section when scores are None."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X[:, 0] + rng.standard_normal(100) * 0.1

        result = compute_rf_importance(
            X,
            y,
            ["a", "b", "c"],
            target_name="target",
            interaction_enabled=False,
        )

        print_rf_table(result)
        captured = capsys.readouterr()

        assert "Pairwise Interaction Scores (Friedman H-statistic)" not in captured.out


class TestComputeInteractionScores:
    """Tests for the _compute_interaction_scores function."""

    def test_symmetric_matrix(self) -> None:
        """Interaction matrix should be symmetric."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 4))
        y = X[:, 0] + X[:, 1] * 0.5 + rng.standard_normal(200) * 0.1

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

        model = RandomForestRegressor(n_estimators=30, random_state=42)
        model.fit(X, y)

        interactions = _compute_interaction_scores(
            model, X, y, [f"f{i}" for i in range(25)]
        )

        assert interactions is None

    def test_non_uniform_values_with_signal(self) -> None:
        """Synergistic feature pairs should have higher H-statistic than non-interacting ones."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 500, 3
        X = rng.standard_normal((n_samples, n_features))
        # Target driven by feature 0 and a synergy between 0 and 1.
        y = (
            X[:, 0] * 2
            + (X[:, 0] * X[:, 1]) * 1.5
            + rng.standard_normal(n_samples) * 0.3
        )

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

        cfg = OmegaConf.create(
            {
                "predict": {"n_history_steps": 3, "n_forecast_steps": 14},
            }
        )

        history, forecast = _get_rf_window_params(cfg)

        assert history == 3
        assert forecast == 14

    def test_rfcfg_overrides_predict(self) -> None:
        """RF config values should override predict config."""
        from omegaconf import OmegaConf  # noqa: PLC0415

        cfg = OmegaConf.create(
            {
                "predict": {"n_history_steps": 3, "n_forecast_steps": 14},
                "rf": {"n_history_steps": 5, "n_forecast_steps": 7},
            }
        )

        history, forecast = _get_rf_window_params(cfg)

        assert history == 5
        assert forecast == 7

    def test_predict_defaults_when_no_n_steps(self) -> None:
        """Should use hardcoded defaults when predict config lacks n_steps."""
        from omegaconf import OmegaConf  # noqa: PLC0415

        cfg = OmegaConf.create(
            {
                "predict": {},
            }
        )

        history, forecast = _get_rf_window_params(cfg)

        assert history == 3
        assert forecast == 14


class TestWindowsToArray:
    """Tests for the _windows_to_arrays function."""

    def test_basic_conversion(self) -> None:
        """Should convert windows to correct X and y shapes."""
        rng = np.random.default_rng(42)
        n_windows, n_history, n_vars = 10, 3, 4

        windows = []
        base_date = np.datetime64("2020-01-01", "D")
        for i in range(n_windows):
            history_features = {
                "group_a": rng.standard_normal((n_history, n_vars)),
            }
            target_value = float(rng.integers(0, 100))
            windows.append(
                RFWindow(
                    start_date=base_date + i,
                    history_features=history_features,
                    target_value=target_value,
                )
            )

        feature_names = [f"v{i}" for i in range(n_vars)]

        X, y, expanded_names = _windows_to_arrays(windows, feature_names, n_history)

        assert X.shape == (n_windows, n_history * n_vars)
        assert y.shape == (n_windows,)
        # Expanded names should have one unique label per column.
        assert len(expanded_names) == n_history * n_vars
        for day_idx in range(n_history):
            for i in range(n_vars):
                suffix = f"t-{n_history - 1 - day_idx}"
                expected = f"{feature_names[i]}_{suffix}"
                assert expanded_names[day_idx * n_vars + i] == expected

    def test_sentinel_values_match_group_variable_and_lag_labels(self) -> None:
        """Every day-major feature value must retain its exact column label."""
        window = RFWindow(
            start_date=np.datetime64("2020-01-01", "D"),
            history_features={
                "sic": np.array([[11.0], [21.0]]),
                "era5": np.array([[12.0, 13.0], [22.0, 23.0]]),
            },
            target_value=0.0,
        )

        X, _, names = _windows_to_arrays(
            [window], ["sic/ice_conc", "era5/t2m", "era5/msl"], 2
        )

        assert names == [
            "sic/ice_conc_t-1",
            "sic/ice_conc_t-0",
            "era5/t2m_t-1",
            "era5/msl_t-1",
            "era5/t2m_t-0",
            "era5/msl_t-0",
        ]
        np.testing.assert_array_equal(X, [[11.0, 21.0, 12.0, 13.0, 22.0, 23.0]])

    def test_target_values_preserved(self) -> None:
        """Target values should match input windows."""
        rng = np.random.default_rng(42)
        n_windows, n_history, n_vars = 5, 2, 3

        target_values = [float(i * 10) for i in range(n_windows)]
        base_date = np.datetime64("2020-01-01", "D")
        windows = []
        for tv_idx, tv in enumerate(target_values):
            history_features = {
                "g": rng.standard_normal((n_history, n_vars)),
            }
            windows.append(
                RFWindow(
                    start_date=base_date + tv_idx,
                    history_features=history_features,
                    target_value=tv,
                )
            )

        feature_names = [f"v{i}" for i in range(n_vars)]

        _, y, expanded_names = _windows_to_arrays(windows, feature_names, n_history)

        np.testing.assert_array_almost_equal(y, target_values)
        assert len(expanded_names) == n_history * n_vars

    def test_empty_windows_raises(self) -> None:
        """Should raise when no windows provided."""
        with pytest.raises(ValueError, match="No windows"):
            _windows_to_arrays([], [], n_history_steps=3)


class TestBuildRFWindows:
    """Tests for build_rf_windows — synthetic data exercises the window logic."""

    def test_empty_datasets_raises(self) -> None:
        """Should raise when no input datasets provided."""
        with pytest.raises(ValueError, match="No input datasets"):
            build_rf_windows({}, None, "ice_conc")  # type: ignore[arg-type]

    def test_missing_history_timestamp_rejects_window(self) -> None:
        """A calendar gap must not be replaced by the next positional sample."""
        dates = [
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-03"),
            np.datetime64("2020-01-04"),
        ]
        values = {date: np.full((1, 1, 1), index) for index, date in enumerate(dates)}
        dataset = FakeSingleDataset("sic", ["ice_conc"], values)

        with pytest.raises(ValueError, match="No valid windows"):
            build_rf_windows(
                {"sic": dataset},  # type: ignore[dict-item]
                dataset,  # type: ignore[arg-type]
                "ice_conc",
                n_history_steps=2,
                n_forecast_steps=1,
            )

    def test_absent_target_variable_raises_clearly(self) -> None:
        """The RF target must never fall back to another channel or their mean."""
        dates = [np.datetime64("2020-01-01") + offset for offset in range(3)]
        values = {date: np.ones((1, 1, 1)) for date in dates}
        dataset = FakeSingleDataset("sic", ["other"], values)

        with pytest.raises(ValueError, match="Configured target variable 'ice_conc'"):
            build_rf_windows(
                {"sic": dataset},  # type: ignore[dict-item]
                dataset,  # type: ignore[arg-type]
                "ice_conc",
                n_history_steps=1,
                n_forecast_steps=1,
            )
