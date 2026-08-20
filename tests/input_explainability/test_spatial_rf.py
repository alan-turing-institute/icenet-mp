"""Unit tests for opt-in sampled-map RF screening."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from icenet_mp.input_explainability.rf import run_rf_analysis
from icenet_mp.input_explainability.spatial_rf import (
    SpatialRFResult,
    SpatialSamples,
    _max_features_fraction,
    _permute_date_blocks,
    build_spatial_samples,
    make_regressor,
    run_spatial_rf,
    save_spatial_rf_results,
)


class FakeSpatialDataset:
    """In-memory TCHW data source with the SingleDataset RF interface."""

    def __init__(
        self, variables: list[str], values: dict[np.datetime64, np.ndarray]
    ) -> None:
        """Initialise the data source."""
        self.variable_names = variables
        self._values = values
        self.dates = sorted(values)
        self.frequency = np.timedelta64(1, "D")

    def get_tchw(self, dates: list[np.datetime64]) -> np.ndarray:
        """Return requested TCHW slices."""
        return np.stack([self._values[date] for date in dates])

    def subset(self, *, date_ranges: list[dict[str, str]]) -> FakeSpatialDataset:
        """Record the requested analysis period and preserve this in-memory source."""
        self.date_ranges = date_ranges
        return self


def _datasets() -> tuple[dict[str, FakeSpatialDataset], FakeSpatialDataset]:
    dates = [np.datetime64("2020-01-01") + day for day in range(8)]
    sic_values = {
        date: np.array([[[0.0, 0.5], [0.9, 0.2]]], dtype=float) + index * 0.01
        for index, date in enumerate(dates)
    }
    weather_values = {
        date: np.array([[[10.0 + index, 20.0], [30.0, 40.0]]], dtype=float)
        for index, date in enumerate(dates)
    }
    sic = FakeSpatialDataset(["ice_conc"], sic_values)
    weather = FakeSpatialDataset(["t2m"], weather_values)
    return {"sic-ssmis": sic, "era5": weather}, sic


def test_spatial_samples_preserve_local_values_and_labels() -> None:
    """Rows must retain local group/day/variable values rather than map means."""
    datasets, target = _datasets()
    samples = build_spatial_samples(
        datasets,  # type: ignore[arg-type]
        target,  # type: ignore[arg-type]
        "ice_conc",
        np.ones((2, 2), dtype=bool),
        n_history_steps=2,
        n_forecast_steps=1,
        seed=7,
        locations_per_stratum=1,
        max_initialisations=1,
    )

    assert samples.feature_names == [
        "sic-ssmis/ice_conc_t-1",
        "sic-ssmis/ice_conc_t-0",
        "era5/t2m_t-1",
        "era5/t2m_t-0",
    ]
    assert samples.features.shape == (3, 4)
    assert set(samples.strata) == {"open_water", "marginal_ice", "pack_ice"}


def test_sampling_does_not_depend_on_future_target_values() -> None:
    """Changing future SIC must not change locations selected at initialisation."""
    datasets, target = _datasets()
    kwargs = {
        "target_variable": "ice_conc",
        "valid_mask": np.ones((2, 2), dtype=bool),
        "n_history_steps": 2,
        "n_forecast_steps": 1,
        "seed": 11,
        "locations_per_stratum": 1,
        "max_initialisations": 1,
    }
    original = build_spatial_samples(datasets, target, **kwargs)  # type: ignore[arg-type]
    future_date = np.datetime64("2020-01-03")
    target._values[future_date] = np.full((1, 2, 2), 0.99)
    changed = build_spatial_samples(datasets, target, **kwargs)  # type: ignore[arg-type]

    np.testing.assert_array_equal(original.features, changed.features)
    np.testing.assert_array_equal(original.strata, changed.strata)


def test_sic_change_targets_subtract_latest_historical_sic_for_every_lead() -> None:
    """Change targets use the initialisation-time SIC at each sampled location."""
    datasets, target = _datasets()
    samples = build_spatial_samples(
        datasets,  # type: ignore[arg-type]
        target,  # type: ignore[arg-type]
        "ice_conc",
        np.ones((2, 2), dtype=bool),
        n_history_steps=2,
        n_forecast_steps=3,
        seed=7,
        locations_per_stratum=1,
        max_initialisations=1,
        target_mode="sic_change",
    )

    latest_sic = samples.features[
        :, samples.feature_names.index("sic-ssmis/ice_conc_t-0")
    ]
    for lead in range(1, 4):
        np.testing.assert_allclose(samples.targets[lead], 0.01 * lead)
        np.testing.assert_allclose(
            samples.targets[lead] + latest_sic, latest_sic + 0.01 * lead
        )
    assert samples.metadata["target_mode"] == "sic_change"


def test_land_cells_are_never_sampled() -> None:
    """The static valid-ocean mask excludes land regardless of SIC values."""
    datasets, target = _datasets()
    mask = np.array([[True, False], [True, True]])
    samples = build_spatial_samples(
        datasets,  # type: ignore[arg-type]
        target,  # type: ignore[arg-type]
        "ice_conc",
        mask,
        n_history_steps=2,
        n_forecast_steps=1,
        seed=7,
        locations_per_stratum=4,
        max_initialisations=1,
    )

    assert samples.features.shape[0] == 3
    assert samples.metadata["valid_ocean_cells"] == 3


def test_non_finite_rows_are_excluded() -> None:
    """Rows containing NaN/inf must be rejected and accounted for."""
    datasets, target = _datasets()
    datasets["era5"]._values[np.datetime64("2020-01-01")][0, 0, 0] = np.nan
    samples = build_spatial_samples(
        datasets,  # type: ignore[arg-type]
        target,  # type: ignore[arg-type]
        "ice_conc",
        np.ones((2, 2), dtype=bool),
        n_history_steps=2,
        n_forecast_steps=1,
        seed=7,
        locations_per_stratum=1,
        max_initialisations=1,
    )

    assert samples.metadata["exclusions"]["non_finite"] == 1


def test_missing_history_timestamp_rejects_window() -> None:
    """A missing intermediate timestamp cannot be replaced by positional data."""
    datasets, target = _datasets()
    missing = np.datetime64("2020-01-02")
    for dataset in datasets.values():
        del dataset._values[missing]
        dataset.dates.remove(missing)

    with pytest.raises(ValueError, match="No complete spatial RF"):
        build_spatial_samples(
            datasets,  # type: ignore[arg-type]
            target,  # type: ignore[arg-type]
            "ice_conc",
            np.ones((2, 2), dtype=bool),
            n_history_steps=2,
            n_forecast_steps=6,
            seed=7,
            locations_per_stratum=1,
        )


def test_initialisation_limit_spans_the_available_training_period() -> None:
    """A bounded budget samples chronologically across, rather than only from, early dates."""
    datasets, target = _datasets()
    samples = build_spatial_samples(
        datasets,  # type: ignore[arg-type]
        target,  # type: ignore[arg-type]
        "ice_conc",
        np.ones((2, 2), dtype=bool),
        n_history_steps=2,
        n_forecast_steps=1,
        seed=7,
        locations_per_stratum=1,
        max_initialisations=2,
    )

    assert samples.metadata["initialisation_count"] == 2
    assert samples.metadata["date_range"] == ["2020-01-01", "2020-01-06"]


def test_group_permutation_moves_complete_initialisation_blocks() -> None:
    """Grouped permutation preserves each date's complete spatial row block."""
    features = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    dates = np.array(
        ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"], dtype="datetime64[D]"
    )

    permuted = _permute_date_blocks(features, dates, [0], np.random.default_rng(0))

    assert set(permuted[:2, 0]) in ({1.0, 2.0}, {3.0, 4.0})
    assert set(permuted[2:, 0]) in ({1.0, 2.0}, {3.0, 4.0})
    np.testing.assert_array_equal(permuted[:, 1], features[:, 1])


def test_per_lead_rf_reports_grouped_importance_and_outputs(
    tmp_path: Path,
) -> None:
    """Per-lead results retain grouped fold distributions and complete reports."""
    dates = [np.datetime64("2020-01-01") + day for day in range(18)]
    base = np.array([[0.0, 0.5], [0.9, 0.2]])
    signal = np.sin(np.arange(len(dates)) * np.pi / 2) * 0.1
    sic_values = {
        date: (base + signal[index])[np.newaxis] for index, date in enumerate(dates)
    }
    weather_values = {
        date: np.full((1, 2, 2), signal[min(index + 1, len(dates) - 1)])
        for index, date in enumerate(dates)
    }
    sic = FakeSpatialDataset(["ice_conc"], sic_values)
    weather = FakeSpatialDataset(["t2m"], weather_values)
    samples = build_spatial_samples(
        {"sic-ssmis": sic, "era5": weather},  # type: ignore[arg-type,dict-item]
        sic,  # type: ignore[arg-type]
        "ice_conc",
        np.ones((2, 2), dtype=bool),
        n_history_steps=2,
        n_forecast_steps=1,
        seed=7,
        locations_per_stratum=1,
    )

    result = run_spatial_rf(
        samples,
        "sic-ssmis",
        "ice_conc",
        n_estimators=20,
        max_depth=5,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        random_state=7,
        n_jobs=1,
        permutation_repeats=2,
        importance_policy="always",
        confirmation_groups=["era5/t2m"],
    )
    json_path, text_path = save_spatial_rf_results(result, tmp_path)

    assert result.leads[0].n_samples > 0
    assert result.leads[0].importance is not None
    weather_importance = result.leads[0].importance["era5/t2m"]
    assert len(weather_importance["fold_values"]) == 5
    assert weather_importance["positive_folds"] <= 5
    assert 0 <= weather_importance["positive_fold_fraction"] <= 1
    assert weather_importance["reliability"] in {
        "stable",
        "candidate",
        "low_evidence",
    }
    assert "marginal_ice" in weather_importance["by_stratum"]
    assert result.leads[0].sic_history_mse is not None
    assert weather_importance["add_to_sic_gain"] is not None
    assert weather_importance["drop_from_full_loss"] is not None
    assert json_path.exists()
    assert text_path.exists()


def test_always_policy_retains_importance_when_rf_loses_to_persistence() -> None:
    """Screening mode calculates evidence independently of surrogate model skill."""
    dates = np.arange(12).astype("timedelta64[D]") + np.datetime64("2020-01-01")
    samples = SpatialSamples(
        features=np.arange(12, dtype=float).reshape(-1, 1),
        feature_names=["sic-ssmis/ice_conc_t-0"],
        targets={1: np.arange(12, dtype=float)},
        initialisations=dates,
        strata=np.full(12, "marginal_ice"),
        metadata={
            "frequency_ns": int(np.timedelta64(1, "D") / np.timedelta64(1, "ns"))
        },
    )

    class LosingRegressor:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, features: np.ndarray, target: np.ndarray) -> LosingRegressor:
            del features, target
            return self

        def predict(self, features: np.ndarray) -> np.ndarray:
            return np.zeros(len(features))

    def run(policy: str) -> SpatialRFResult:
        return run_spatial_rf(
            samples,
            "sic-ssmis",
            "ice_conc",
            n_estimators=1,
            max_depth=1,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features="sqrt",
            random_state=7,
            n_jobs=1,
            permutation_repeats=1,
            importance_policy=policy,  # type: ignore[arg-type]
        )

    with patch(
        "icenet_mp.input_explainability.spatial_rf.RandomForestRegressor",
        LosingRegressor,
    ):
        qualified = run("qualified")
        screening = run("always")

    assert qualified.leads[0].importance_interpretable is False
    assert qualified.leads[0].importance is None
    assert screening.leads[0].importance_interpretable is False
    assert screening.leads[0].importance is not None
    assert screening.metadata["rf_settings"]["importance_policy"] == "always"


def test_sic_change_persistence_baseline_predicts_zero_change() -> None:
    """Persistence means zero change rather than an absolute SIC prediction."""
    dates = np.arange(12).astype("timedelta64[D]") + np.datetime64("2020-01-01")
    samples = SpatialSamples(
        features=np.arange(12, dtype=float).reshape(-1, 1),
        feature_names=["sic-ssmis/ice_conc_t-0"],
        targets={1: np.zeros(12)},
        initialisations=dates,
        strata=np.full(12, "marginal_ice"),
        metadata={
            "frequency_ns": int(np.timedelta64(1, "D") / np.timedelta64(1, "ns")),
            "target_mode": "sic_change",
        },
    )

    result = run_spatial_rf(
        samples,
        "sic-ssmis",
        "ice_conc",
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        random_state=7,
        n_jobs=1,
        permutation_repeats=1,
        importance_policy="always",
    )

    assert result.leads[0].baseline_mse == 0


def test_make_regressor_dispatches_to_random_forest_and_hist_gradient_boosting() -> (
    None
):
    """Each backend builds a regressor with the shared constructor signature."""
    common: dict[str, Any] = {
        "n_estimators": 4,
        "max_depth": 3,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_features": "sqrt",
        "random_state": 0,
        "n_jobs": 1,
        "n_features": 1,
    }
    rf = make_regressor("random_forest", **common)
    hgb = make_regressor("hist_gradient_boosting", **common)

    assert type(rf).__name__ == "RandomForestRegressor"
    assert type(hgb).__name__ == "HistGradientBoostingRegressor"

    features = np.arange(40, dtype=float).reshape(-1, 1)
    target = features.ravel()
    for regressor in (rf, hgb):
        regressor.fit(features, target)
        assert regressor.predict(features).shape == (features.shape[0],)


def test_make_regressor_rejects_unknown_backend() -> None:
    """Unknown backends fail loudly before training starts."""
    with pytest.raises(ValueError, match="Unknown regressor backend"):
        make_regressor(
            "catboost",  # type: ignore[arg-type]
            n_estimators=2,
            max_depth=2,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features="sqrt",
            random_state=0,
            n_jobs=1,
            n_features=1,
        )


def test_max_features_fraction_translates_sqrt_and_log2() -> None:
    """Both string tokens map to fractions in (0, 1] matching scikit-learn semantics."""
    sqrt_value = _max_features_fraction("sqrt", n_features=36)
    log2_value = _max_features_fraction("log2", n_features=36)

    assert sqrt_value == pytest.approx(1.0 / math.sqrt(36))
    assert log2_value == pytest.approx(math.log2(36) / 36)
    assert 0.0 < sqrt_value <= 1.0
    assert 0.0 < log2_value <= 1.0


def test_max_features_fraction_passes_through_numeric_values() -> None:
    """Numeric ``max_features`` values pass through unchanged; ``None`` means all features."""
    assert _max_features_fraction(0.4, n_features=10) == 0.4
    assert _max_features_fraction(None, n_features=10) == 1.0


def test_max_features_fraction_rejects_unknown_strings() -> None:
    """Unknown string tokens raise ``ValueError`` before reaching the backend."""
    with pytest.raises(ValueError, match="Unsupported"):
        _max_features_fraction("auto", n_features=10)


def test_run_spatial_rf_records_backend_choice() -> None:
    """The selected backend is persisted in metadata for downstream consumers."""
    dates = np.arange(36).astype("timedelta64[D]") + np.datetime64("2020-01-01")
    samples = SpatialSamples(
        features=np.arange(72, dtype=float).reshape(36, 2),
        feature_names=["sic-ssmis/ice_conc_t-0", "era5/t2m_t-0"],
        targets={1: np.zeros(36)},
        initialisations=dates,
        strata=np.full(36, "marginal_ice"),
        metadata={
            "frequency_ns": int(np.timedelta64(1, "D") / np.timedelta64(1, "ns")),
            "target_mode": "absolute",
        },
    )

    result = run_spatial_rf(
        samples,
        "sic-ssmis",
        "ice_conc",
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        random_state=7,
        n_jobs=1,
        permutation_repeats=1,
        importance_policy="always",
        backend="hist_gradient_boosting",
    )

    assert result.metadata["rf_settings"]["backend"] == "hist_gradient_boosting"


def test_shared_runner_uses_training_ranges_and_mask(tmp_path: Path) -> None:
    """The shared RF entry point applies training bounds and loads its configured mask."""
    dates = [np.datetime64("2020-01-01") + day for day in range(18)]
    base = np.array([[0.0, 0.5], [0.9, 0.2]])
    sic = FakeSpatialDataset(
        ["ice_conc"],
        {date: (base + index * 0.001)[np.newaxis] for index, date in enumerate(dates)},
    )
    weather = FakeSpatialDataset(
        ["t2m"],
        {date: np.full((1, 2, 2), index % 3) for index, date in enumerate(dates)},
    )
    mask_path = tmp_path / "data/preprocessing/masks/test-sic"
    mask_path.mkdir(parents=True)
    np.save(mask_path / "land_mask.npy", np.ones((2, 2), dtype=bool))
    config = OmegaConf.create(
        {
            "base_path": str(tmp_path),
            "data": {
                "datasets": {},
                "split": {"train": [{"start": "2020-01-01", "end": "2020-01-31"}]},
            },
            "vif": {"max_samples": None},
            "rf": {
                "mode": "spatial",
                "n_history_steps": 2,
                "n_forecast_steps": 1,
                "n_estimators": 10,
                "max_depth": 4,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
                "max_features": "sqrt",
                "random_state": 2,
                "n_jobs": 1,
                "target": {"group_as": "sic-ssmis", "variable": "ice_conc"},
                "spatial": {
                    "mask_dataset_name": "test-sic",
                    "locations_per_stratum": 1,
                    "max_initialisations": 10,
                    "permutation_repeats": 1,
                },
            },
        }
    )
    with (
        patch(
            "icenet_mp.input_diagnostics.data.resolve_datasets", return_value=({}, {})
        ),
        patch(
            "icenet_mp.input_diagnostics.data.build_datasets",
            return_value={"sic-ssmis": sic, "era5": weather},
        ),
    ):
        result = run_rf_analysis(config)

    assert isinstance(result, SpatialRFResult)
    assert sic.date_ranges == [{"start": "2020-01-01", "end": "2020-01-31"}]
    assert result.metadata["mask_identity"].endswith("land_mask.npy")
