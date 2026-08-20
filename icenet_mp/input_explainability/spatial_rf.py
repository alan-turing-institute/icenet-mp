# ruff: noqa: EM101, EM102, PLR2004, TRY003
"""Opt-in sampled-map Random Forest screening for SIC predictors.

This module deliberately keeps spatial screening separate from the scalar RF
diagnostic in ``rf.py``.  Rows are ``(initialisation date, location)`` pairs and
every target lead is evaluated independently.  The implementation uses only
observations at or before the initialisation date for predictors and strata.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from icenet_mp.data_loaders.single_dataset import SingleDataset

RegressorBackend = Literal["random_forest", "hist_gradient_boosting"]


@dataclass(frozen=True)
class SpatialSamples:
    """Local RF rows and provenance produced from temporal map windows."""

    features: np.ndarray
    feature_names: list[str]
    targets: dict[int, np.ndarray]
    initialisations: np.ndarray
    strata: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LeadResult:
    """Held-out skill and grouped permutation importance for one forecast lead."""

    lead: int
    n_samples: int
    mse: float
    mae: float
    baseline_mse: float
    baseline_mae: float
    sic_history_mse: float | None
    by_stratum: dict[str, dict[str, float]]
    importance: dict[str, dict[str, Any]] | None
    importance_interpretable: bool
    fold_boundaries: list[dict[str, str]]


@dataclass(frozen=True)
class SpatialRFResult:
    """Complete result of an opt-in sampled-map RF screening run."""

    feature_names: list[str]
    leads: list[LeadResult]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def _valid_starts(
    datasets: Mapping[str, SingleDataset],
    target_ds: SingleDataset,
    n_history_steps: int,
    n_forecast_steps: int,
) -> tuple[list[np.datetime64], np.timedelta64]:
    """Return starts whose expected history and target timestamps all exist."""
    if not datasets:
        raise ValueError("No input datasets provided for spatial RF screening.")

    frequency = next(iter(datasets.values())).frequency
    common = set.intersection(*(set(dataset.dates) for dataset in datasets.values()))
    target_dates = set(target_ds.dates)
    starts: list[np.datetime64] = []
    for start in sorted(common):
        history = [start + idx * frequency for idx in range(n_history_steps)]
        forecast = [
            start + (n_history_steps + idx) * frequency
            for idx in range(n_forecast_steps)
        ]
        if all(date in common for date in history) and all(
            date in target_dates for date in forecast
        ):
            starts.append(start)
    if not starts:
        raise ValueError(
            "No complete spatial RF history and forecast windows are available."
        )
    return starts, frequency


def _sample_locations(
    sic: np.ndarray,
    valid_mask: np.ndarray,
    *,
    seed: int,
    locations_per_stratum: int,
    open_water_max: float,
    pack_ice_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample valid cells deterministically using initialisation-time SIC only."""
    if sic.shape != valid_mask.shape:
        raise ValueError("The valid-ocean mask must match the target map shape.")
    finite_ocean = valid_mask.astype(bool) & np.isfinite(sic)
    stratum_masks = {
        "open_water": finite_ocean & (sic <= open_water_max),
        "pack_ice": finite_ocean & (sic >= pack_ice_min),
        "marginal_ice": finite_ocean & (sic > open_water_max) & (sic < pack_ice_min),
    }
    rng = np.random.default_rng(seed)
    locations: list[np.ndarray] = []
    labels: list[str] = []
    for name, candidate_mask in stratum_masks.items():
        candidates = np.argwhere(candidate_mask)
        if candidates.size == 0:
            continue
        count = min(locations_per_stratum, len(candidates))
        selected = candidates[rng.choice(len(candidates), size=count, replace=False)]
        locations.append(selected)
        labels.extend([name] * count)
    if not locations:
        raise ValueError(
            "No finite valid-ocean locations were available for spatial RF screening."
        )
    return np.concatenate(locations), np.asarray(labels, dtype="U16")


def build_spatial_samples(  # noqa: C901, PLR0912, PLR0913, PLR0915
    datasets: Mapping[str, SingleDataset],
    target_ds: SingleDataset,
    target_variable: str,
    valid_mask: np.ndarray,
    *,
    n_history_steps: int,
    n_forecast_steps: int,
    seed: int,
    locations_per_stratum: int,
    open_water_max: float = 0.15,
    pack_ice_min: float = 0.8,
    max_initialisations: int | None = None,
    max_rows: int | None = None,
    row_batch_size: int = 1024,
    mask_identity: str | None = None,
    target_mode: Literal["absolute", "sic_change"] = "absolute",
) -> SpatialSamples:
    """Construct finite local-feature rows for each complete temporal window.

    ``valid_mask`` is static.  Sampling strata use the latest history SIC map, never
    a future target map; callers must therefore provide a pre-existing valid-ocean
    mask rather than deriving one from the selected forecast period.
    """
    if target_variable not in target_ds.variable_names:
        raise ValueError(
            f"Configured target variable {target_variable!r} is unavailable."
        )
    if row_batch_size <= 0:
        raise ValueError("row_batch_size must be positive.")
    if target_mode not in {"absolute", "sic_change"}:
        raise ValueError("target_mode must be 'absolute' or 'sic_change'.")
    starts, frequency = _valid_starts(
        datasets, target_ds, n_history_steps, n_forecast_steps
    )
    if max_initialisations is not None:
        if max_initialisations <= 0:
            msg = "max_initialisations must be positive when configured."
            raise ValueError(msg)
        indices = np.linspace(
            0, len(starts) - 1, min(max_initialisations, len(starts)), dtype=int
        )
        starts = [starts[index] for index in indices]

    target_channel = target_ds.variable_names.index(target_variable)
    feature_names = [
        f"{group}/{variable}_t-{n_history_steps - 1 - lag}"
        for group, dataset in datasets.items()
        for lag in range(n_history_steps)
        for variable in dataset.variable_names
    ]
    maximum_rows = len(starts) * 3 * locations_per_stratum
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive when configured.")
        maximum_rows = min(maximum_rows, max_rows)
    feature_matrix = np.empty((maximum_rows, len(feature_names)), dtype=float)
    initialisation_array = np.empty(maximum_rows, dtype="datetime64[ns]")
    strata_array = np.empty(maximum_rows, dtype="U16")
    target_arrays = {
        lead: np.empty(maximum_rows, dtype=float)
        for lead in range(1, n_forecast_steps + 1)
    }
    row_count = 0
    exclusions = {"non_finite": 0, "no_locations": 0}

    for date_index, start in enumerate(starts):
        history_dates = [start + lag * frequency for lag in range(n_history_steps)]
        target_dates = [
            start + (n_history_steps + lag) * frequency
            for lag in range(n_forecast_steps)
        ]
        feature_maps = [
            dataset.get_tchw(history_dates) for dataset in datasets.values()
        ]
        target_maps = target_ds.get_tchw(target_dates)[:, target_channel]
        sic_at_initialisation = target_ds.get_tchw([history_dates[-1]])[
            0, target_channel
        ]
        locations, strata = _sample_locations(
            sic_at_initialisation,
            valid_mask,
            seed=seed + date_index,
            locations_per_stratum=locations_per_stratum,
            open_water_max=open_water_max,
            pack_ice_min=pack_ice_min,
        )
        if len(locations) == 0:
            exclusions["no_locations"] += 1
            continue

        for batch_start in range(0, len(locations), row_batch_size):
            batch_locations = locations[batch_start : batch_start + row_batch_size]
            batch_strata = strata[batch_start : batch_start + row_batch_size]
            for location, stratum in zip(batch_locations, batch_strata, strict=True):
                if row_count == maximum_rows:
                    break
                y_idx, x_idx = location
                values = np.concatenate(
                    [maps[:, :, y_idx, x_idx].reshape(-1) for maps in feature_maps]
                )
                target_values = target_maps[:, y_idx, x_idx]
                if not (np.isfinite(values).all() and np.isfinite(target_values).all()):
                    exclusions["non_finite"] += 1
                    continue
                feature_matrix[row_count] = values
                initialisation_array[row_count] = start
                strata_array[row_count] = str(stratum)
                for lead, value in enumerate(target_values, start=1):
                    target_arrays[lead][row_count] = (
                        value - sic_at_initialisation[y_idx, x_idx]
                        if target_mode == "sic_change"
                        else value
                    )
                row_count += 1
            if row_count == maximum_rows:
                break

    if not row_count:
        raise ValueError("Spatial RF screening rejected every sampled row as invalid.")
    feature_matrix = feature_matrix[:row_count]
    initialisation_array = initialisation_array[:row_count]
    strata_array = strata_array[:row_count]
    if feature_matrix.shape[1] != len(feature_names):
        raise RuntimeError(
            "Spatial RF feature values do not match their generated labels."
        )
    return SpatialSamples(
        features=feature_matrix,
        feature_names=feature_names,
        targets={lead: values[:row_count] for lead, values in target_arrays.items()},
        initialisations=initialisation_array,
        strata=strata_array,
        metadata={
            "seed": seed,
            "target_mode": target_mode,
            "sampling": {
                "max_initialisations": max_initialisations,
                "locations_per_stratum": locations_per_stratum,
                "max_rows": max_rows,
                "row_batch_size": row_batch_size,
                "open_water_max": open_water_max,
                "pack_ice_min": pack_ice_min,
            },
            "mask_shape": list(valid_mask.shape),
            "valid_ocean_cells": int(valid_mask.astype(bool).sum()),
            "mask_identity": mask_identity or "caller-supplied",
            "stratum_counts": {
                name: int((strata_array == name).sum())
                for name in ("open_water", "pack_ice", "marginal_ice")
            },
            "effective_samples": row_count,
            "initialisation_count": len(set(initialisation_array)),
            "exclusions": exclusions,
            "date_range": [
                np.datetime_as_string(initialisation_array.min(), unit="D"),
                np.datetime_as_string(initialisation_array.max(), unit="D"),
            ],
            "frequency_ns": int(frequency / np.timedelta64(1, "ns")),
        },
    )


def _group_columns(feature_names: Sequence[str]) -> dict[str, list[int]]:
    """Group all lags of a physical variable under one name."""
    result: dict[str, list[int]] = {}
    for index, name in enumerate(feature_names):
        group = name.rsplit("_t-", maxsplit=1)[0]
        result.setdefault(group, []).append(index)
    return result


def _max_features_fraction(value: str | float | None, *, n_features: int) -> float:
    """Translate scikit-learn ``max_features`` to a fraction.

    ``HistGradientBoostingRegressor`` only accepts a float in ``(0, 1]`` (it has no
    ``"sqrt"`` / ``"log2"`` tokens); we therefore expand those tokens using
    ``n_features`` so the scikit-learn ``RandomForestRegressor`` semantics are
    preserved: ``"sqrt"`` selects ``sqrt(n_features)`` columns per split (fraction
    ``1 / sqrt(n_features)``), ``"log2"`` selects ``log2(n_features)`` columns per
    split (fraction ``log2(n_features) / n_features``).
    """
    if value is None:
        return 1.0
    if isinstance(value, str):
        if value == "sqrt":
            return math.sqrt(1.0 / n_features)
        if value == "log2":
            return math.log2(n_features) / n_features
        msg = (
            f"Unsupported HistGradientBoostingRegressor max_features value: {value!r}."
        )
        raise ValueError(msg)
    if not 0.0 < value <= 1.0:
        msg = "HistGradientBoostingRegressor max_features must lie in (0, 1]."
        raise ValueError(msg)
    return value


def make_regressor(  # noqa: PLR0913
    backend: RegressorBackend,
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    min_samples_split: int,
    max_features: str | float | None,
    random_state: int,
    n_jobs: int,
    n_features: int,
) -> Any:  # noqa: ANN401
    """Construct the configured tree-based regressor with shared hyperparameters.

    ``hist_gradient_boosting`` uses scikit-learn's own
    ``HistGradientBoostingRegressor``, which shares scikit-learn's bundled OpenMP
    runtime rather than linking a second, independently-packaged one (unlike the
    LightGBM/XGBoost wheels this backend replaced, which each ship a build
    linked against a separate Homebrew ``libomp.dylib`` on macOS and reliably
    segfault when used alongside scikit-learn in the same process). It has no
    ``min_samples_split`` or row-subsampling equivalent (ignored, matching
    ``random_forest``'s ``min_samples_split`` handling by the prior gradient-
    boosted backends) and no per-estimator ``n_jobs`` knob (``n_jobs`` is
    ignored; threading is controlled by scikit-learn's global thread pool).
    ``n_features`` is required to translate the scikit-learn ``"sqrt"`` /
    ``"log2"`` ``max_features`` tokens into the float fraction
    ``HistGradientBoostingRegressor`` expects.
    """
    if backend == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,  # type: ignore[arg-type]
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if backend == "hist_gradient_boosting":
        feature_fraction = _max_features_fraction(max_features, n_features=n_features)
        return HistGradientBoostingRegressor(
            max_iter=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=max(1, int(min_samples_leaf)),
            max_features=feature_fraction,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=random_state,
        )
    msg = f"Unknown regressor backend: {backend!r}."
    raise ValueError(msg)


def _rank_stability(fold_importance: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Return each group's normalised stability of rank across temporal folds."""
    groups = list(fold_importance)
    if len(groups) < 2:
        return dict.fromkeys(groups, 1.0)
    values = np.asarray([fold_importance[group] for group in groups], dtype=float)
    ranks = np.empty_like(values)
    for fold_index in range(values.shape[1]):
        order = np.argsort(-values[:, fold_index], kind="stable")
        ranks[order, fold_index] = np.arange(len(groups))
    maximum_std = (len(groups) - 1) / 2
    return {
        group: float(max(0.0, 1.0 - np.std(ranks[index]) / maximum_std))
        for index, group in enumerate(groups)
    }


def _importance_summary(
    values: Sequence[float],
    rank_stability: float,
    *,
    stable_positive_fraction: float,
    stable_rank_stability: float,
) -> dict[str, Any]:
    """Summarise fold consistency without implying causal importance."""
    positive_folds = sum(value > 0 for value in values)
    positive_fraction = positive_folds / len(values) if values else 0.0
    mean = float(np.mean(values)) if values else 0.0
    if mean <= 0:
        reliability = "low_evidence"
    elif (
        positive_fraction >= stable_positive_fraction
        and rank_stability >= stable_rank_stability
    ):
        reliability = "stable"
    else:
        reliability = "candidate"
    return {
        "mean_mse_increase": mean,
        "std_mse_increase": float(np.std(values)) if values else 0.0,
        "fold_values": [float(value) for value in values],
        "positive_folds": positive_folds,
        "positive_fold_fraction": positive_fraction,
        "rank_stability": rank_stability,
        "reliability": reliability,
    }


def _permute_date_blocks(
    features: np.ndarray,
    initialisations: np.ndarray,
    columns: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Jointly permute whole initialisation-date blocks without breaking row alignment."""
    result = features.copy()
    date_rows = [
        np.flatnonzero(initialisations == date) for date in np.unique(initialisations)
    ]
    by_size: dict[int, list[np.ndarray]] = {}
    for rows in date_rows:
        by_size.setdefault(len(rows), []).append(rows)
    for row_blocks in by_size.values():
        if len(row_blocks) < 2:
            continue
        order = rng.permutation(len(row_blocks))
        column_indices = np.asarray(columns)
        for target_index, source_index in enumerate(order):
            result[np.ix_(row_blocks[target_index], column_indices)] = features[
                np.ix_(row_blocks[source_index], column_indices)
            ]
    return result


def _fold_indices(
    initialisations: np.ndarray, lead: int, frequency: np.timedelta64
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create chronological date-level folds and purge overlapping target windows."""
    unique_dates = np.unique(initialisations)
    if len(unique_dates) < 6:
        raise ValueError(
            "Spatial RF screening requires at least six initialisation dates."
        )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_dates_idx, test_dates_idx in TimeSeriesSplit(n_splits=5).split(
        unique_dates
    ):
        train_dates = unique_dates[train_dates_idx]
        test_dates = unique_dates[test_dates_idx]
        first_test = test_dates.min()
        # A training start has target date start + lead; purge overlap with the first
        # validation target interval conservatively at date granularity.
        train_dates = train_dates[train_dates + lead * frequency < first_test]
        train_rows = np.flatnonzero(np.isin(initialisations, train_dates))
        test_rows = np.flatnonzero(np.isin(initialisations, test_dates))
        if len(train_rows) and len(test_rows):
            folds.append((train_rows, test_rows))
    if not folds:
        raise ValueError(
            "Temporal purging left no train/validation folds for spatial RF screening."
        )
    return folds


def run_spatial_rf(  # noqa: C901, PLR0912, PLR0913, PLR0915
    samples: SpatialSamples,
    target_group: str,
    target_variable: str,
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    min_samples_split: int,
    max_features: str | float | None,
    random_state: int,
    n_jobs: int,
    permutation_repeats: int = 10,
    confirmation_groups: Sequence[str] = (),
    feature_groups: Mapping[str, Sequence[int]] | None = None,
    importance_policy: Literal["qualified", "always"] = "qualified",
    stable_positive_fraction: float = 0.8,
    stable_rank_stability: float = 0.5,
    backend: RegressorBackend = "random_forest",
) -> SpatialRFResult:
    """Fit one temporally validated tree-based regressor per lead.

    The default ``backend='random_forest'`` preserves the original scikit-learn
    behaviour. ``hist_gradient_boosting`` is a drop-in alternative with shared
    permutation-importance and confirmation-refit semantics.
    """
    if backend not in {"random_forest", "hist_gradient_boosting"}:
        raise ValueError("backend must be 'random_forest' or 'hist_gradient_boosting'.")
    if importance_policy not in {"qualified", "always"}:
        raise ValueError("importance_policy must be 'qualified' or 'always'.")
    if not 0 <= stable_positive_fraction <= 1:
        raise ValueError("stable_positive_fraction must be between zero and one.")
    if not 0 <= stable_rank_stability <= 1:
        raise ValueError("stable_rank_stability must be between zero and one.")
    baseline_name = f"{target_group}/{target_variable}_t-0"
    try:
        baseline_column = samples.feature_names.index(baseline_name)
    except ValueError as exc:
        raise ValueError(
            "Spatial RF screening requires latest historical target SIC as a feature."
        ) from exc
    groups = (
        {group: list(columns) for group, columns in feature_groups.items()}
        if feature_groups is not None
        else _group_columns(samples.feature_names)
    )
    results: list[LeadResult] = []
    show_progress = sys.stderr.isatty()

    for lead, y in tqdm(
        samples.targets.items(),
        total=len(samples.targets),
        desc="Spatial RF leads",
        disable=not show_progress,
    ):
        fold_importance: dict[str, list[float]] = {group: [] for group in groups}
        fold_stratum_importance: dict[str, dict[str, list[float]]] = {
            group: {} for group in groups
        }
        y_true: list[float] = []
        y_pred: list[float] = []
        baseline_pred: list[float] = []
        stratum_values: list[str] = []
        frequency = np.timedelta64(int(samples.metadata["frequency_ns"]), "ns")
        folds = _fold_indices(samples.initialisations, lead, frequency)
        fold_boundaries: list[dict[str, str]] = []
        full_fold_mse: list[float] = []
        for fold_index, (train_idx, test_idx) in enumerate(
            tqdm(
                folds,
                total=len(folds),
                desc=f"Lead {lead} folds",
                leave=False,
                disable=not show_progress,
            )
        ):
            fold_boundaries.append(
                {
                    "train_start": str(samples.initialisations[train_idx].min()),
                    "train_end": str(samples.initialisations[train_idx].max()),
                    "validation_start": str(samples.initialisations[test_idx].min()),
                    "validation_end": str(samples.initialisations[test_idx].max()),
                }
            )
            model = make_regressor(
                backend,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                max_features=max_features,
                random_state=random_state + fold_index,
                n_jobs=n_jobs,
                n_features=samples.features.shape[1],
            )
            model.fit(samples.features[train_idx], y[train_idx])
            test_features = samples.features[test_idx]
            prediction = model.predict(test_features)
            y_true.extend(y[test_idx])
            y_pred.extend(prediction)
            baseline_pred.extend(
                np.zeros(len(test_idx))
                if samples.metadata.get("target_mode", "absolute") == "sic_change"
                else test_features[:, baseline_column]
            )
            stratum_values.extend(samples.strata[test_idx])

            baseline_fold_mse = mean_squared_error(y[test_idx], prediction)
            full_fold_mse.append(float(baseline_fold_mse))
            fold_strata = samples.strata[test_idx]
            for group, columns in groups.items():
                scores: list[float] = []
                stratum_scores: dict[str, list[float]] = {
                    stratum: [] for stratum in np.unique(fold_strata)
                }
                for repeat in range(permutation_repeats):
                    permuted = _permute_date_blocks(
                        test_features,
                        samples.initialisations[test_idx],
                        columns,
                        np.random.default_rng(
                            random_state + fold_index * 1000 + repeat
                        ),
                    )
                    permuted_prediction = model.predict(permuted)
                    scores.append(
                        mean_squared_error(y[test_idx], permuted_prediction)
                        - baseline_fold_mse
                    )
                    for stratum, values in stratum_scores.items():
                        selected = fold_strata == stratum
                        values.append(
                            float(
                                mean_squared_error(
                                    y[test_idx][selected], permuted_prediction[selected]
                                )
                                - mean_squared_error(
                                    y[test_idx][selected], prediction[selected]
                                )
                            )
                        )
                fold_importance[group].append(float(np.mean(scores)))
                for stratum, values in stratum_scores.items():
                    fold_stratum_importance[group].setdefault(str(stratum), []).append(
                        float(np.mean(values))
                    )

        y_true_array = np.asarray(y_true)
        y_pred_array = np.asarray(y_pred)
        baseline_array = np.asarray(baseline_pred)
        mse = float(mean_squared_error(y_true_array, y_pred_array))
        mae = float(mean_absolute_error(y_true_array, y_pred_array))
        baseline_mse = float(mean_squared_error(y_true_array, baseline_array))
        baseline_mae = float(mean_absolute_error(y_true_array, baseline_array))
        by_stratum: dict[str, dict[str, float]] = {}
        stratum_array = np.asarray(stratum_values)
        for stratum in np.unique(stratum_array):
            selected = stratum_array == stratum
            by_stratum[str(stratum)] = {
                "n_samples": float(selected.sum()),
                "mse": float(
                    mean_squared_error(y_true_array[selected], y_pred_array[selected])
                ),
                "mae": float(
                    mean_absolute_error(y_true_array[selected], y_pred_array[selected])
                ),
                "baseline_mse": float(
                    mean_squared_error(y_true_array[selected], baseline_array[selected])
                ),
                "baseline_mae": float(
                    mean_absolute_error(
                        y_true_array[selected], baseline_array[selected]
                    )
                ),
                "mse_improvement": float(
                    mean_squared_error(y_true_array[selected], baseline_array[selected])
                    - mean_squared_error(y_true_array[selected], y_pred_array[selected])
                ),
            }
        interpretable = mse < baseline_mse
        retain_importance = interpretable or importance_policy == "always"
        rank_stability = _rank_stability(fold_importance)
        stratum_rank_stability = {
            stratum: _rank_stability(
                {
                    group: values_by_stratum[stratum]
                    for group, values_by_stratum in fold_stratum_importance.items()
                    if stratum in values_by_stratum
                }
            )
            for stratum in sorted(
                {
                    stratum
                    for values_by_stratum in fold_stratum_importance.values()
                    for stratum in values_by_stratum
                }
            )
        }
        importance = (
            {
                group: {
                    **_importance_summary(
                        values,
                        rank_stability[group],
                        stable_positive_fraction=stable_positive_fraction,
                        stable_rank_stability=stable_rank_stability,
                    ),
                    "by_stratum": {
                        stratum: _importance_summary(
                            stratum_values,
                            stratum_rank_stability[stratum][group],
                            stable_positive_fraction=stable_positive_fraction,
                            stable_rank_stability=stable_rank_stability,
                        )
                        for stratum, stratum_values in fold_stratum_importance[
                            group
                        ].items()
                    },
                }
                for group, values in fold_importance.items()
            }
            if retain_importance
            else None
        )
        sic_history_mse: float | None = None
        if importance is not None and confirmation_groups:
            sic_history_columns = groups.get(
                f"{target_group}/{target_variable}", [baseline_column]
            )
            sic_history_scores: list[float] = []
            for fold_index, (train_idx, test_idx) in enumerate(folds):
                sic_model = make_regressor(
                    backend,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    random_state=random_state + fold_index,
                    n_jobs=n_jobs,
                    n_features=len(sic_history_columns),
                )
                sic_model.fit(
                    samples.features[train_idx][:, sic_history_columns], y[train_idx]
                )
                sic_history_scores.append(
                    float(
                        mean_squared_error(
                            y[test_idx],
                            sic_model.predict(
                                samples.features[test_idx][:, sic_history_columns]
                            ),
                        )
                    )
                )
            sic_history_mse = float(np.mean(sic_history_scores))
            for group in confirmation_groups:
                confirmation_columns = groups.get(group)
                if confirmation_columns is None:
                    continue
                drop_columns = [
                    index
                    for index in range(samples.features.shape[1])
                    if index not in confirmation_columns
                ]
                add_columns = sorted(
                    set(confirmation_columns) | set(sic_history_columns)
                )
                drop_mse: list[float] = []
                add_mse: list[float] = []
                for fold_index, (train_idx, test_idx) in enumerate(folds):
                    for selected_columns, scores in (
                        (drop_columns, drop_mse),
                        (add_columns, add_mse),
                    ):
                        if not selected_columns:
                            continue
                        model = make_regressor(
                            backend,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            min_samples_split=min_samples_split,
                            max_features=max_features,
                            random_state=random_state + fold_index,
                            n_jobs=n_jobs,
                            n_features=len(selected_columns),
                        )
                        model.fit(
                            samples.features[train_idx][:, selected_columns],
                            y[train_idx],
                        )
                        scores.append(
                            float(
                                mean_squared_error(
                                    y[test_idx],
                                    model.predict(
                                        samples.features[test_idx][:, selected_columns]
                                    ),
                                )
                            )
                        )
                importance[group]["drop_group_mse"] = (
                    float(np.mean(drop_mse)) if drop_mse else None
                )
                importance[group]["add_group_mse"] = (
                    float(np.mean(add_mse)) if add_mse else None
                )
                importance[group]["add_to_sic_gain"] = (
                    sic_history_mse - float(np.mean(add_mse))
                    if add_mse and sic_history_mse is not None
                    else None
                )
                importance[group]["drop_from_full_loss"] = (
                    float(np.mean(drop_mse)) - float(np.mean(full_fold_mse))
                    if drop_mse
                    else None
                )
        results.append(
            LeadResult(
                lead=lead,
                n_samples=len(y_true),
                mse=mse,
                mae=mae,
                baseline_mse=baseline_mse,
                baseline_mae=baseline_mae,
                sic_history_mse=sic_history_mse,
                by_stratum=by_stratum,
                importance=importance,
                importance_interpretable=interpretable,
                fold_boundaries=fold_boundaries,
            )
        )
    metadata = {
        **samples.metadata,
        "analysis_mode": "sampled_map_screening",
        "target": f"{target_group}/{target_variable}",
        "feature_groups": groups,
        "rf_settings": {
            "backend": backend,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "permutation_repeats": permutation_repeats,
            "confirmation_groups": list(confirmation_groups),
            "importance_policy": importance_policy,
            "reliability_thresholds": {
                "stable_positive_fraction": stable_positive_fraction,
                "stable_rank_stability": stable_rank_stability,
            },
        },
    }
    return SpatialRFResult(samples.feature_names, results, metadata)


def save_spatial_rf_results(
    result: SpatialRFResult, output_dir: Path
) -> tuple[Path, Path]:
    """Write complete machine-readable and human-readable spatial screening reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "spatial_rf_results.json"
    json_path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    lines = ["Sampled-map Random Forest Screening", "=" * 40, ""]
    lines.append(
        "Importance is predictive, model-specific and non-causal; correlated groups may substitute."
    )
    lines.append(f"Regressor backend: {result.metadata['rf_settings']['backend']}")
    lines.append(
        f"Importance policy: {result.metadata['rf_settings']['importance_policy']}"
    )
    lines.append(f"Effective samples: {result.metadata['effective_samples']}")
    date_range = result.metadata["date_range"]
    lines.append(f"Date range: {date_range[0]} to {date_range[1]}")
    lines.append("")
    for lead in result.leads:
        lines.extend(
            [
                f"Lead {lead.lead}",
                f"  RF MSE/MAE: {lead.mse:.6f} / {lead.mae:.6f}",
                f"  Persistence MSE/MAE: {lead.baseline_mse:.6f} / {lead.baseline_mae:.6f}",
            ]
        )
        if lead.importance is None:
            lines.append(
                "  Importance not retained: RF did not beat persistence on held-out data."
            )
        else:
            if not lead.importance_interpretable:
                lines.append(
                    "  Model-quality context: RF did not beat persistence; importance is retained for exploratory screening."
                )
            lines.append(
                "  Grouped permutation importance (mean MSE increase ± fold standard deviation):"
            )
            for group, score in sorted(
                lead.importance.items(), key=lambda item: -item[1]["mean_mse_increase"]
            ):
                lines.append(
                    f"    {group}: {score['mean_mse_increase']:.6f} ± {score['std_mse_increase']:.6f}"
                )
    text_path = output_dir / "spatial_rf_report.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, text_path
