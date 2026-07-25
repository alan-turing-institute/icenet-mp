"""Random Forest feature importance for input variable explainability.

Trains a Random Forest regressor to predict the target variable (e.g., next-day SIC)
from all other input variables, then derives per-feature importance from permutation-
based scores on a held-out test set. Also computes pairwise interaction strengths.

**Temporal windows:** Each RF sample is built from a window of ``n_history_steps`` days
of input history followed by ``n_forecast_steps`` forecast days for the target. This
matches how the model is trained (via ``CombinedDataset``), replacing the old single-day
sampling approach that ignored temporal structure.

**Interpretation:**
- **Permutation importance**: How much does the model's MSE increase when this feature
  is randomly shuffled? A large increase means the model relies heavily on this feature
  for accurate predictions — it has high predictive contribution to the target.
- **Interaction scores**: How much more (or less) does performance degrade when two
  features are shuffled together versus individually? Positive = synergistic interaction;
  negative = redundant / overlapping information.

Unlike diagnostics (VIF, PCA, EOF), this is **supervised** — it measures how inputs
actually contribute to predicting the target variable, not just their variance structure.

"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig  # noqa: TC002
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from icenet_mp.data_loaders.single_dataset import SingleDataset


logger = logging.getLogger(__name__)

# Maximum number of features for interaction heatmap (computationally expensive).
_MAX_INTERACTION_FEATURES = 20


@dataclass(frozen=True)
class RFWindow:
    """A single temporal window used as one RF sample."""

    start_date: np.datetime64
    history_features: dict[str, np.ndarray]  # group_name → (n_history_steps, n_variables_in_group) spatial means
    target_value: float


def _get_rf_window_params(config: DictConfig) -> tuple[int, int]:
    """Resolve ``n_history_steps`` and ``n_forecast_steps`` for RF analysis.

    Prefers explicit ``rf.n_history_steps`` / ``rf.n_forecast_steps`` from the RF config;
    falls back to ``config.predict.*`` when using baseline configs that set these via
    the predict group (e.g. ``sic-ssmis-14d`` → history=3, forecast=14).

    Args:
        config: Hydra-composed config.

    Returns:
        Tuple of ``(n_history_steps, n_forecast_steps)``.

    """
    rf_cfg = config.get("rf", {}) or {}
    n_history = rf_cfg.get("n_history_steps")
    n_forecast = rf_cfg.get("n_forecast_steps")

    # Fall back to predict group if not explicitly set in RF config.
    if n_history is None:
        predict_cfg = config.get("predict", {}) or {}
        n_history = predict_cfg.get("n_history_steps", 3)
    if n_forecast is None:
        predict_cfg = config.get("predict", {}) or {}
        n_forecast = predict_cfg.get("n_forecast_steps", 14)

    return int(n_history), int(n_forecast)


def build_rf_windows(  # noqa: C901, PLR0912, PLR0915 — data-loading function with many steps
    datasets: dict[str, SingleDataset],
    target_path: Path | None = None,
    target_variable: str | None = None,
    *,
    n_history_steps: int = 3,
    n_forecast_steps: int = 14,
    max_samples: int | None = None,
) -> tuple[list[RFWindow], list[str]]:
    """Build temporal windows for Random Forest samples.

    Mirrors ``CombinedDataset.dates`` logic: finds start dates where all required
    history and forecast timesteps are available across the relevant datasets.

    For each valid window:
    - **Features**: spatial mean of each input variable per history day → flattened to
      one row per window (shape: ``n_history_steps x n_variables``).
    - **Target**: spatial mean of the target variable averaged across all forecast steps.

    Args:
        datasets: Mapping of group name to SingleDataset for input features.
        target_path: Path to the zarr store containing the target variable.
            When ``None``, the first dataset in *datasets* is used as the target source.
        target_variable: Name of the target variable within the zarr store or dataset.
            When ``None``, uses the first variable from the target dataset.
        n_history_steps: Number of past days to use as input history.
        n_forecast_steps: Number of future days to average for the target.
        max_samples: Maximum number of windows to return (chronological order).

    Returns:
        Tuple of ``(windows, feature_names)`` where ``windows`` is a list of
        :class:`RFWindow` objects and ``feature_names`` is a flat list of column labels
        in the order they appear in each window's flattened features.

    Raises:
        ValueError: If no valid windows can be built or if the dataset has too few dates.

    """
    import zarr  # noqa: PLC0415

    n_input_datasets = len(datasets)
    if n_input_datasets == 0:
        msg = "No input datasets provided for RF window building."
        raise ValueError(msg)

    # Determine target source: explicit path or fall back to first dataset.
    use_target_dataset = target_path is None and target_variable is None
    if use_target_dataset:
        assert datasets  # guaranteed by check above  # noqa: S101
        target_ds = next(iter(datasets.values()))
    else:
        target_ds = None

    # Step 1: Find dates common to ALL input feature datasets.
    input_date_sets = [set(ds.dates) for ds in datasets.values()]
    common_input_dates = sorted(set.intersection(*input_date_sets))

    if len(common_input_dates) < n_history_steps:
        msg = (
            f"Not enough common dates ({len(common_input_dates)}) across input datasets "
            f"to support {n_history_steps}-day history windows. "
            "Reduce n_history_steps or use a dataset with more overlapping dates."
        )
        raise ValueError(msg)

    # Step 2: Load target dates.
    if use_target_dataset:
        assert target_ds is not None  # mypy narrowing  # noqa: S101
        # Target is in one of the input datasets — use its date list.
        target_dates = sorted(target_ds.dates)
        target_date_set = set(target_dates)
        logger.info("Using first dataset %r as target source (%d dates).", target_ds.name, len(target_dates))
    else:
        # Target is in a separate zarr store.
        if not target_path or not target_variable:
            msg = "Both target_path and target_variable must be provided when not using an input dataset as target."
            raise ValueError(msg)

        store = zarr.DirectoryStore(str(target_path))
        root = zarr.group(store=store)

        if target_variable not in root:
            available_vars = list(root.keys())
            msg = (
                f"Target variable {target_variable!r} not found in zarr at {target_path}. "
                f"Available variables: {available_vars}"
            )
            raise ValueError(msg)

        target_dates = sorted([np.datetime64(d, "D") for d in root.attrs.get("dates", [])])
        if not target_dates:
            msg = f"No dates found in target zarr {target_path}."
            raise ValueError(msg)

        target_date_set = set(target_dates)

    # Step 3: Build feature name list (order must match window construction).
    feature_names = [
        f"{group_name}/{var_name}" for group_name, ds in datasets.items() for var_name in ds.variable_names
    ]

    if not feature_names:
        msg = "No variables found in any input dataset."
        raise ValueError(msg)

    # Step 4: Find valid window start dates.
    # A start date is valid when:
    #   - All n_history_steps history dates exist in ALL input datasets (already guaranteed
    #     by common_input_dates, since those are the intersection).
    #   - All n_forecast_steps forecast dates exist in the target dataset.
    frequency = next(iter(datasets.values())).frequency  # all have same frequency

    valid_starts: list[np.datetime64] = []
    for start_date in common_input_dates:
        forecast_dates = [
            start_date + (idx + n_history_steps) * frequency
            for idx in range(n_forecast_steps)
        ]
        if all(fd in target_date_set for fd in forecast_dates):
            valid_starts.append(start_date)

    if not valid_starts:
        msg = (
            f"No valid windows found. "
            f"Input datasets share {len(common_input_dates)} dates, but none have "
            f"{n_forecast_steps} consecutive forecast dates available in the target."
        )
        raise ValueError(msg)

    logger.info(
        "Found %d valid RF windows (history=%d, forecast=%d).",
        len(valid_starts), n_history_steps, n_forecast_steps,
    )

    # Step 5: Apply max_samples limit chronologically.
    if max_samples is not None and max_samples < len(valid_starts):
        indices = np.linspace(0, len(valid_starts) - 1, max_samples, dtype=int)
        valid_starts = [valid_starts[i] for i in indices]
        logger.info("Sampling %d windows evenly from %d available.", max_samples, len(valid_starts))

    # Step 6: Build window data.
    windows: list[RFWindow] = []

    # Determine target variable name for the dataset path.
    if use_target_dataset:
        assert target_ds is not None  # mypy narrowing  # noqa: S101
        target_var_name = str(target_ds.variable_names[0]) if target_variable is None else target_variable
    else:
        target_var_name = str(target_variable)  # type: ignore[assignment]

    for start_date in valid_starts:
        history_features: dict[str, np.ndarray] = {}
        try:
            for group_name, ds in datasets.items():
                tchw = ds.get_tchw_slice(start_date, n_history_steps)  # (n_history_steps, C, H, W)
                spatial_means = tchw.mean(axis=(-2, -1))  # (n_history_steps, C)
                history_features[group_name] = spatial_means
        except Exception:  # noqa: BLE001 — MissingDateError when intermediate dates are absent
            logger.warning("Skipping window starting %s: missing data in history.", start_date)
            continue

        # Target: spatial mean of target variable averaged across all forecast steps.
        forecast_dates = [
            start_date + (idx + n_history_steps) * frequency
            for idx in range(n_forecast_steps)
        ]
        target_values: list[float] = []

        if use_target_dataset:
            assert target_ds is not None  # mypy narrowing  # noqa: S101
            # Target is in a SingleDataset — use get_tchw_slice.
            for fd in forecast_dates:
                try:
                    tchw = target_ds.get_tchw_slice(fd, 1)  # (T=1, C, H, W)
                    # Find the channel index for the target variable.
                    if target_var_name in target_ds.variable_names:
                        ch_idx = target_ds.variable_names.index(target_var_name)
                        spatial_mean = float(tchw[0, ch_idx].mean())
                    else:
                        spatial_mean = float(tchw.mean())
                    target_values.append(spatial_mean)
                except (KeyError, ValueError):
                    logger.warning("Target date %s not found in dataset; skipping window.", fd)
        else:
            # Target is in a zarr store — read directly.
            for fd in forecast_dates:
                date_str = str(fd)[:10]  # "YYYY-MM-DD"
                if target_var_name in root and date_str in root[target_var_name]:
                    var_data = root[target_var_name][date_str][:]
                    if var_data.ndim >= 2:  # noqa: PLR2004 — check for spatial dimensions
                        target_values.append(float(var_data.mean()))
                    else:
                        target_values.append(float(var_data))

        if not target_values:
            logger.warning("No target data for window starting %s; skipping.", start_date)
            continue

        target_value = float(np.mean(target_values))
        windows.append(RFWindow(start_date=start_date, history_features=history_features, target_value=target_value))

    logger.info(
        "Built %d RF windows with %d features each (total feature vector: %d).",
        len(windows), len(feature_names), len(feature_names) * n_history_steps if n_history_steps > 1 else len(feature_names),
    )

    return windows, feature_names


def _windows_to_arrays(
    windows: list[RFWindow],
    feature_names: list[str],
    n_history_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of RFWindows to feature matrix X and target vector y.

    Each window's features are flattened across history steps and variables:
    ``[day0_var0, day0_var1, ..., day1_var0, ...]`` giving
    ``n_history_steps x n_variables`` columns per sample.

    Args:
        windows: List of RFWindow objects.
        feature_names: Flat list of variable names (one per column per history step).
        n_history_steps: Number of history steps used in each window.

    Returns:
        Tuple of ``(X, y)`` where X has shape ``(n_windows, n_features_per_window)``
        and y has shape ``(n_windows,)``.

    """
    if not windows:
        msg = "No windows to convert."
        raise ValueError(msg)

    n_windows = len(windows)
    n_vars = len(feature_names)  # variables per history step
    n_features_per_window = n_vars * n_history_steps

    X = np.zeros((n_windows, n_features_per_window))  # noqa: N806 — standard ML convention for feature matrix
    y = np.zeros(n_windows)

    for i, window in enumerate(windows):
        col = 0
        # spatial_means shape: (n_history_steps, n_vars_in_group)
        for spatial_means in window.history_features.values():
            for day_idx in range(spatial_means.shape[0]):
                X[i, col:col + spatial_means.shape[1]] = spatial_means[day_idx]
                col += spatial_means.shape[1]

        y[i] = window.target_value

    return X, y


@dataclass(frozen=True)
class RFResult:
    """Holds the results of a Random Forest explainability run."""

    feature_names: list[str]
    target_name: str
    n_samples: int
    n_features: int
    permutation_importance: np.ndarray  # (n_features,) — mean across CV folds
    importances_std: np.ndarray  # (n_features,) — std across CV folds
    r2_score: float
    mse: float
    interaction_scores: np.ndarray | None = None  # (n_features, n_features) or None


def _compute_interaction_scores(
    model: RandomForestRegressor,
    X: np.ndarray,  # noqa: N803 — standard ML convention for feature matrix
    y: np.ndarray,  # noqa: ARG001 — unused; kept for API consistency with callers
    _feature_names: Sequence[str],
    max_features: int = _MAX_INTERACTION_FEATURES,
    grid_resolution: int = 20,
) -> np.ndarray | None:
    """Compute pairwise Friedman H-statistic interaction scores.

    For each pair (i, j), the H-statistic measures deviation from additivity in
    partial dependence space [Friedman1997]_.  It is defined as:

        H(i,j) = sqrt( var(f_ij - f_i - f_j + mu) / var(f_ij) )

    where ``f_ij`` is the joint partial dependence surface, ``f_i`` and ``f_j`` are
    the individual surfaces, and ``mu`` is their overall mean.  Values range from 0
    (purely additive — no interaction) to 1 (maximal interaction).

    **Interpretation:**

    - **Near zero** → *additive*: features contribute independently to predictions.
    - **High (≥ 0.3)** → *strong interaction*: the pair's combined effect deviates
      significantly from a simple sum of individual effects.  This may indicate
      synergy, redundancy, or non-linear coupling — further investigation is needed.

    The H-statistic is preferred over permutation-based interaction scores because it
    operates in partial dependence space where additive decomposition is well-defined,
    avoiding the sign ambiguities of joint-permutation approaches.

    Args:
        model: Trained RandomForestRegressor.
        X: Feature matrix, shape ``(n_samples, n_features)``.
        y: Target values, shape ``(n_samples,)`` (unused; kept for API consistency).
        feature_names: One name per column in X (unused; kept for API consistency).
        max_features: Maximum number of features to include in interaction analysis.
        grid_resolution: Number of grid points per axis for partial dependence grids.

    Returns:
        Symmetric interaction matrix, shape ``(n_features, n_features)``, or None if
        too many features (computationally expensive: O(n_features²) PD evaluations).

    .. _Friedman1997:
       https://statweb.stanford.edu/~jhf/ftp/trebst.pdf

    """
    from sklearn.inspection import partial_dependence  # noqa: PLC0415

    n_features = X.shape[1]

    if n_features > max_features:
        logger.warning(
            "Skipping interaction analysis: %d features exceeds limit of %d. "
            "Interaction scores will be None.",
            n_features,
            max_features,
        )
        return None

    interactions = np.zeros((n_features, n_features))

    total_pairs = n_features * (n_features - 1) // 2
    logger.info("Computing %d Friedman H-statistic interaction scores...", total_pairs)

    for i in range(n_features):
        pd_i = partial_dependence(
            model, X, [i], grid_resolution=grid_resolution, method="brute",
        ).average.flatten()  # shape: (G,)

        for j in range(i + 1, n_features):
            pd_joint = partial_dependence(
                model, X, [i, j], grid_resolution=grid_resolution, method="brute",
            ).average[0]  # shape: (G, G)

            pd_j = partial_dependence(
                model, X, [j], grid_resolution=grid_resolution, method="brute",
            ).average.flatten()  # shape: (G,)

            overall_mean = np.mean(pd_joint)

            # Interaction effect: joint PD minus additive decomposition.
            # For purely additive features: f_ij(a,b) ≈ f_i(a) + f_j(b) - mu
            pd_individual = pd_i[:, np.newaxis] + pd_j[np.newaxis, :] - overall_mean
            interaction_effect = pd_joint - pd_individual

            var_interaction = float(np.var(interaction_effect))
            var_total = float(np.var(pd_joint))

            h_squared = var_interaction / var_total if var_total > 0 else 0.0
            interactions[i, j] = np.sqrt(h_squared)
            interactions[j, i] = interactions[i, j]  # symmetric

    return interactions


def compute_rf_importance(  # noqa: PLR0913 — standard ML function with many hyperparams
    X: np.ndarray,  # noqa: N803 — standard ML convention for feature matrix
    y: np.ndarray,
    feature_names: Sequence[str],
    target_name: str = "target",
    *,
    n_estimators: int = 500,
    max_depth: int | None = 15,
    min_samples_leaf: int = 5,
    min_samples_split: int = 8,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RFResult:
    """Train a Random Forest and compute permutation-based feature importance.

    Uses **time-series cross-validation** (consecutive folds, no shuffling) to compute
    per-fold permutation importances, then averages across folds for a robust estimate.
    Also computes R² and MSE on held-out test folds.

    TimeSeriesSplit is used instead of KFold because RF samples are temporal windows —
    shuffling would leak future information into the training set (a sample's history
    includes dates before its start, so random splits would train on data from after
    the test date).

    **Hyperparameter defaults are tuned for small-to-moderate datasets** (hundreds to
    thousands of samples) common in climate/geoscience analysis.  ``max_depth``,
    ``min_samples_leaf``, and ``max_features`` work together to prevent overfitting —
    with unlimited depth on ~1000 samples the model memorises training noise, which
    makes permutation importances collapse to near-zero for all features.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Target values, shape (n_samples,).
        feature_names: One name per column in X.
        target_name: Human-readable name of the predicted target variable.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth (None = unlimited; default 15 for small data).
        min_samples_leaf: Minimum samples required at each leaf node.
        min_samples_split: Minimum samples required to split an internal node.
        max_features: Number of features to consider per split ("sqrt", "log2", float, int).
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 = all CPUs).

    Returns:
        RFResult with importance scores, fit metrics, and metadata.

    """
    n_samples, n_features = X.shape
    tscv = TimeSeriesSplit(n_splits=5)

    importances_all: list[np.ndarray] = []
    r2_scores: list[float] = []
    mse_values: list[float] = []
    oob_scores: list[float] = []
    trained_models: list[RandomForestRegressor] = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state + fold_idx,
            n_jobs=n_jobs,
            oob_score=True,
        )
        model.fit(x_train, y_train)
        trained_models.append(model)

        # Evaluate on test fold.
        y_pred = model.predict(x_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        mse = float(np.mean((y_test - y_pred) ** 2))

        # Out-of-bag score as an additional generalisation check.
        oob_scores.append(float(model.oob_score_))

        # Permutation importance on test fold (more repeats for stable estimates).
        imp = permutation_importance(
            model, x_test, y_test, n_repeats=20, random_state=random_state,
            scoring="neg_mean_squared_error", n_jobs=n_jobs,
        )
        importances_all.append(imp.importances_mean)

        r2_scores.append(r2)
        mse_values.append(mse)

        logger.info(
            "Fold %d: R²=%.4f, MSE=%.6f, OOB=%.4f", fold_idx + 1, r2, mse, model.oob_score_,
        )

    # Average importances and metrics across folds.
    fold_permutation_imp = np.mean(importances_all, axis=0)
    importances_std = np.std(importances_all, axis=0)
    mean_r2 = float(np.mean(r2_scores))
    mean_mse = float(np.mean(mse_values))
    mean_oob = float(np.mean(oob_scores))

    logger.info(
        "RF importance complete: %d samples, %d features, "
        "mean R²=%.4f, MSE=%.6f, OOB=%.4f.",
        n_samples, n_features, mean_r2, mean_mse, mean_oob,
    )

    # Compute interaction scores (only if feature count is manageable).
    final_model = trained_models[0]  # use first fold's model for interactions
    interaction_scores = _compute_interaction_scores(
        final_model, X, y, feature_names, max_features=_MAX_INTERACTION_FEATURES,
    )

    return RFResult(
        feature_names=list(feature_names),
        target_name=target_name,
        n_samples=n_samples,
        n_features=n_features,
        permutation_importance=fold_permutation_imp,
        importances_std=importances_std,
        r2_score=mean_r2,
        mse=mean_mse,
        interaction_scores=interaction_scores,
    )


def run_rf_analysis(  # noqa: C901 — complex data-loading function
    config: DictConfig,
) -> RFResult:
    """Run a full Random Forest explainability analysis from a Hydra-composed config.

    Resolves dataset paths and variable selections for input features, loads the target
    variable (SIC/ice_conc), builds **temporal windows** matching how the model is trained
    (via ``CombinedDataset``), and computes permutation-based importance with interaction
    analysis.

    Each RF sample is a window of ``n_history_steps`` days of input history followed by
    ``n_forecast_steps`` forecast days for the target. This replaces the old single-day
    sampling approach that ignored temporal structure.

    Args:
        config: Hydra-composed config (from ``imp rf`` or ``imp input-explainability run``).

    Returns:
        RFResult with computed scores.

    """
    from icenet_mp.input_diagnostics.data import (  # noqa: PLC0415
        _get_max_samples,
        build_datasets,
        resolve_datasets,
    )

    rf_cfg = config.get("rf", {}) or {}
    max_samples = _get_max_samples(config, "rf")

    # Resolve temporal window parameters.
    n_history_steps, n_forecast_steps = _get_rf_window_params(config)
    logger.info(
        "RF temporal windows: history=%d steps, forecast=%d steps.",
        n_history_steps, n_forecast_steps,
    )

    # Resolve input feature datasets.
    group_paths, group_variables = resolve_datasets(config)
    datasets = build_datasets(group_paths, group_variables)

    # Load target variable (SIC/ice_conc).
    target_cfg = rf_cfg.get("target", {}) or {}
    target_group_as = str(target_cfg.get("group_as", "sic-ssmis"))
    target_variable = str(target_cfg.get("variable", "ice_conc"))

    logger.info(
        "Loading target variable %r from group %r.",
        target_variable, target_group_as,
    )

    # Find the zarr file for this group — look up by resolved group name first.
    target_path: Path | None = None
    if target_group_as in group_paths:
        paths_for_group = group_paths[target_group_as]
        if paths_for_group:
            target_path = paths_for_group[0]

    # Fallback: scan for any zarr whose path contains the target group name.
    if target_path is None:
        for paths in group_paths.values():
            for p in paths:
                if target_group_as in str(p):
                    target_path = p
                    break
            if target_path is not None:
                break

    if target_path is None:
        msg = (
            f"Target dataset {target_group_as}.zarr not found. "
            f"Ensure the configured datasets include a group matching '{target_group_as}'."
        )
        raise ValueError(msg)

    # Build temporal windows — this handles date intersection, max_samples limiting,
    # and window construction in one step.
    windows, var_names = build_rf_windows(
        datasets, target_path, target_variable,
        n_history_steps=n_history_steps,
        n_forecast_steps=n_forecast_steps,
        max_samples=max_samples,
    )

    logger.info("Built %d RF windows with %d features per window.", len(windows), len(var_names))

    # Convert windows to feature matrix and target vector.
    X, y = _windows_to_arrays(windows, var_names, n_history_steps)  # noqa: N806 — standard ML convention for feature matrix

    logger.info(
        "Final arrays — features: %s, target (%s): %s", X.shape, target_variable, y.shape,
    )

    # Prevent data leakage: if the target variable is also present as a feature column,
    # remove all columns corresponding to it across all history steps.  This can happen
    # when the SIC dataset appears in both the input datasets list and as the target group
    # (the common case for explainability).
    target_label = f"{target_group_as}/{target_variable}"
    leak_indices: list[int] = []
    new_var_names: list[str] = []

    n_vars_per_step = len(var_names)  # variables per history step
    total_feature_cols = n_vars_per_step * n_history_steps

    for col in range(total_feature_cols):
        var_idx_in_step = col % n_vars_per_step
        candidate_name = var_names[var_idx_in_step] if var_idx_in_step < len(var_names) else ""
        if candidate_name == target_label:
            leak_indices.append(col)
        else:
            new_var_names.append(candidate_name)

    if leak_indices:
        logger.info(
            "Removing %d data-leakage column(s) for feature %r across all history steps.",
            len(leak_indices), target_label,
        )
        X = np.delete(X, leak_indices, axis=1)  # noqa: N806 — standard ML convention for feature matrix

    return compute_rf_importance(
        X, y, new_var_names, target_name=f"{target_group_as}/{target_variable}",
        n_estimators=rf_cfg.get("n_estimators", 500),
        max_depth=rf_cfg.get("max_depth", 15),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
        min_samples_split=rf_cfg.get("min_samples_split", 8),
        max_features=rf_cfg.get("max_features", "sqrt"),
        random_state=rf_cfg.get("random_state", 42),
        n_jobs=rf_cfg.get("n_jobs", -1),
    )


def print_rf_table(result: RFResult) -> None:
    """Print a formatted RF results table to stdout.

    Args:
        result: RFResult from ``compute_rf_importance`` or ``run_rf_analysis``.

    """
    names = result.feature_names
    importance = result.permutation_importance
    std = result.importances_std

    print(  # noqa: T201
        f"\nRandom Forest Feature Importance — Predicting {result.target_name}"
    )
    print("-" * 75)  # noqa: T201

    # Model fit summary.
    print("\nModel Fit (Time-series 5-fold CV):")  # noqa: T201
    print(f"  R²   = {result.r2_score:.4f}")  # noqa: T201
    print(f"  MSE  = {result.mse:.6f}")  # noqa: T201
    print(f"  Samples = {result.n_samples}, Features = {result.n_features}")  # noqa: T201

    # Feature importance ranking.
    print("\nFeature Importance (permutation, mean MSE increase):")  # noqa: T201
    print("-" * 75)  # noqa: T201
    order = np.argsort(-importance)
    print(f"{'Rank':<5} {'Variable':<45} {'Importance':>12} {'Std':>10}")  # noqa: T201
    print("-" * 75)  # noqa: T201

    for rank, idx in enumerate(order, start=1):
        flag = " ***" if importance[idx] > 3 * np.mean(np.abs(importance)) else ""
        print(  # noqa: T201
            f"{rank:<5} {names[idx]:<45} {importance[idx]:>12.6f} {std[idx]:>10.6f}{flag}"
        )

    print()  # noqa: T201


def save_rf_results(result: RFResult, output_dir: Path) -> tuple[Path, Path, list[Path]]:
    """Save RF results to JSON and a text report in the given directory.

    Also generates visualisation plots (feature importance bar chart + interaction heatmap).

    Args:
        result: RFResult from ``compute_rf_importance`` or ``run_rf_analysis``.
        output_dir: Directory to write files into (created if missing).

    Returns:
        Tuple of (json_path, txt_path, plot_paths) where plot_paths is a list of saved PNGs.

    """
    from .base import plot_feature_importance, plot_interaction_heatmap  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report — machine-readable, includes all metadata.
    json_path = output_dir / "rf_results.json"
    serialisable = asdict(result)
    serialisable["permutation_importance"] = result.permutation_importance.tolist()
    serialisable["importances_std"] = result.importances_std.tolist()
    if result.interaction_scores is not None:
        serialisable["interaction_scores"] = result.interaction_scores.tolist()
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)

    # Text report — human-readable table.
    txt_path = output_dir / "rf_report.txt"
    lines: list[str] = []
    lines.append(f"Random Forest Feature Importance — Predicting {result.target_name}")
    lines.append("-" * 75)
    lines.append("")
    lines.append("Model Fit (Time-series 5-fold CV):")
    lines.append(f"  R²   = {result.r2_score:.4f}")
    lines.append(f"  MSE  = {result.mse:.6f}")
    lines.append(f"  Samples = {result.n_samples}, Features = {result.n_features}")
    lines.append("")
    lines.append("Feature Importance (permutation, mean MSE increase):")
    lines.append("-" * 75)

    names = result.feature_names
    importance = result.permutation_importance
    std = result.importances_std
    order = np.argsort(-importance)

    for rank, idx in enumerate(order, start=1):
        flag = " ***" if importance[idx] > 3 * np.mean(np.abs(importance)) else ""
        lines.append(
            f"{rank:<5} {names[idx]:<45} {importance[idx]:>12.6f} {std[idx]:>10.6f}{flag}"
        )

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # Generate plots.
    plot_paths: list[Path] = []
    fig_path = plot_feature_importance(result, output_dir)  # type: ignore[arg-type]
    plot_paths.append(fig_path)
    logger.info("Feature importance plot written to %s.", fig_path)

    heat_path = plot_interaction_heatmap(result, output_dir)  # type: ignore[arg-type]
    if heat_path is not None:
        plot_paths.append(heat_path)
        logger.info("Interaction heatmap written to %s.", heat_path)

    logger.info("RF results written to %s and %s.", json_path, txt_path)
    return json_path, txt_path, plot_paths
