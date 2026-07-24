"""Random Forest feature importance for input variable explainability.

Trains a Random Forest regressor to predict the target variable (e.g., next-day SIC)
from all other input variables, then derives per-feature importance from permutation-
based scores on a held-out test set. Also computes pairwise interaction strengths.

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
from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)

# Maximum number of features for interaction heatmap (computationally expensive).
_MAX_INTERACTION_FEATURES = 20


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

    Uses k-fold cross-validation to compute per-fold permutation importances, then
    averages across folds for a robust estimate. Also computes R² and MSE on held-out
    test folds.

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
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    importances_all: list[np.ndarray] = []
    r2_scores: list[float] = []
    mse_values: list[float] = []
    oob_scores: list[float] = []
    trained_models: list[RandomForestRegressor] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
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


def run_rf_analysis(  # noqa: C901, PLR0912, PLR0915 — complex data-loading function
    config: DictConfig,
) -> RFResult:
    """Run a full Random Forest explainability analysis from a Hydra-composed config.

    Resolves dataset paths and variable selections for input features, loads the target
    variable (SIC/ice_conc), constructs feature + target arrays, and computes permutation-
    based importance with interaction analysis.

    Args:
        config: Hydra-composed config (from ``imp rf`` or ``imp input-explainability run``).

    Returns:
        RFResult with computed scores.

    """
    from icenet_mp.input_diagnostics.data import (  # noqa: PLC0415
        _get_max_samples,
        build_datasets,
        build_sample_matrix,
        resolve_datasets,
    )

    rf_cfg = config.get("rf", {})
    max_samples = _get_max_samples(config, "rf")

    # Resolve input feature datasets.
    group_paths, group_variables = resolve_datasets(config)
    datasets = build_datasets(group_paths, group_variables)

    sample_matrix, var_names = build_sample_matrix(datasets, max_samples=max_samples)

    logger.info("Feature matrix shape: %s (%d features).", sample_matrix.shape, len(var_names))

    # Load target variable (SIC/ice_conc).
    target_cfg = rf_cfg.get("target", {})
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

    # Load target from zarr — we need to read it directly since SingleDataset expects
    # a group name that matches the dataset's internal naming.
    import zarr  # noqa: PLC0415

    store = zarr.DirectoryStore(str(target_path))
    root = zarr.group(store=store)

    # Find the target variable in the zarr structure.
    if target_variable not in root:
        available_vars = list(root.keys())
        msg = (
            f"Target variable {target_variable!r} not found in zarr. "
            f"Available variables: {available_vars}"
        )
        raise ValueError(msg)

    # Get all dates from the target dataset for alignment.
    target_dates = sorted([np.datetime64(d, "D") for d in root.attrs.get("dates", [])])

    if not target_dates:
        msg = f"No dates found in target zarr {target_path}."
        raise ValueError(msg)

    # Intersect with feature dates.
    # We need to re-build the sample matrix with date intersection against target dates.
    # First, get all dataset dates for intersection.
    feature_datasets = datasets
    common_dates = sorted(set.intersection(*(set(ds.dates) for ds in feature_datasets.values())))
    target_date_set = set(target_dates)

    aligned_dates = sorted(set(common_dates) & target_date_set)

    if not aligned_dates:
        msg = (
            f"No common dates between features and target {target_variable!r}. "
            f"Features cover {len(common_dates)} dates, target covers {len(target_dates)}."
        )
        raise ValueError(msg)

    # Apply max_samples limit.
    if max_samples is not None and max_samples < len(aligned_dates):
        indices = np.linspace(0, len(aligned_dates) - 1, max_samples, dtype=int)
        aligned_dates = [aligned_dates[i] for i in indices]
        logger.info("Aligned %d dates; sampling %d evenly.", len(common_dates), max_samples)

    # Build feature matrix from aligned dates.
    var_names_out: list[str] = []
    feature_arrays: list[np.ndarray] = []

    for group_name, ds in feature_datasets.items():
        tchw = ds.get_tchw(aligned_dates)  # (T, C, H, W)
        for var_idx, var_name in enumerate(ds.variable_names):
            full_label = f"{group_name}/{var_name}"
            var_names_out.append(full_label)
            c = tchw.shape[1]
            if var_idx < c:
                spatial_mean = tchw[:, var_idx, :, :].mean(axis=(-2, -1))
                feature_arrays.append(spatial_mean)

    X = np.column_stack(feature_arrays)  # noqa: N806 — standard ML convention for feature matrix

    # Build target array from aligned dates.
    target_array = []
    for date in aligned_dates:
        date_str = str(date)[:10]  # "YYYY-MM-DD"
        if date_str in root[target_variable]:
            var_data = root[target_variable][date_str][:]
            # Spatial mean of target (same as feature aggregation).
            if var_data.ndim >= 2:  # noqa: PLR2004 — check for spatial dimensions
                target_array.append(var_data.mean())
            else:
                target_array.append(float(var_data))

    y = np.array(target_array)

    logger.info(
        "Final arrays — features: %s, target (%s): %s", X.shape, target_variable, y.shape,
    )

    # Prevent data leakage: if the target variable is also present as a feature column,
    # remove it from X.  This can happen when the SIC dataset appears in both the input
    # datasets list and as the target group (the common case for explainability).
    target_label = f"{target_group_as}/{target_variable}"
    if target_label in var_names_out:
        leak_idx = var_names_out.index(target_label)
        logger.info(
            "Removing data-leakage feature %r from X (it is also the prediction target).",
            target_label,
        )
        X = np.delete(X, leak_idx, axis=1)  # noqa: N806 — standard ML convention for feature matrix
        var_names_out.pop(leak_idx)

    return compute_rf_importance(
        X, y, var_names_out, target_name=f"{target_group_as}/{target_variable}",
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
    print("\nModel Fit (5-fold CV):")  # noqa: T201
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
    lines.append("Model Fit (5-fold CV):")
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
