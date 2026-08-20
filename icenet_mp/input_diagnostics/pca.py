"""PCA (Principal Component Analysis) for input variable feature importance.

Standardises variables, computes principal components via SVD, and derives per-variable
importance scores from weighted absolute loadings across all components.

**Interpretation:**
- Each PC captures a direction of maximum variance in the input data.
- A variable with high **weighted loading** loads strongly onto PCs that explain lots
  of variance — it carries more information content than low-scoring variables.
- The cumulative explained variance tells you how many components capture most of the
  signal (e.g., "7 components explain 90% of input variance").

PCA is **unsupervised**: it measures variance in input variables alone, not predictive
power for SIC. A high-importance variable may be redundant with others (check VIF), or
it may carry unique information that happens to vary a lot across timesteps.

Use PCA alongside VIF: if two variables both score high on the same PC and have high
VIF, they are likely capturing the same underlying signal — keep one.

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_MINIMUM_VARIABLES = 2

# Threshold for cumulative explained variance (components needed to capture 90%).
VARIANCE_THRESHOLD = 0.90


@dataclass(frozen=True)
class PCAResult:
    """Holds the results of a PCA analysis run."""

    variable_names: list[str]
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    components: np.ndarray  # (n_components, n_features)
    feature_importance: np.ndarray  # weighted absolute loadings per variable
    n_samples: int
    n_features: int


def compute_pca(
    sample_matrix: np.ndarray,
    variable_names: list[str],
) -> PCAResult:
    """Compute PCA and derive feature importance from component loadings.

    Args:
        sample_matrix: Array of shape ``(n_samples, n_variables)`` — already spatially
            aggregated, one row per timestep.
        variable_names: One name per column.

    Returns:
        PCAResult with scores, components, and metadata.

    Raises:
        ValueError: If fewer than 2 variables are provided (PCA requires at least two
            variables to compute principal components).

    """
    n_vars = sample_matrix.shape[1]
    if n_vars < _MINIMUM_VARIABLES:
        msg = f"PCA requires at least 2 variables to compute principal components; got {n_vars}."
        logger.warning(msg)
        raise ValueError(msg)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(sample_matrix)

    pca = PCA()
    pca.fit(x_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # Components: shape (n_components, n_features). Transpose to (n_features, n_components).
    loadings = np.abs(pca.components_.T)  # (n_features, n_components)

    # Feature importance: weighted sum of absolute loadings across all components.
    feature_importance = loadings @ explained_variance_ratio  # (n_features,)

    logger.info(
        "PCA complete: %d samples, %d features, top component explains %.1f%% variance.",
        sample_matrix.shape[0],
        len(variable_names),
        explained_variance_ratio[0] * 100,
    )

    return PCAResult(
        variable_names=variable_names,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_explained_variance=cumulative_explained_variance,
        components=pca.components_,
        feature_importance=feature_importance,
        n_samples=sample_matrix.shape[0],
        n_features=len(variable_names),
    )


def print_pca_table(result: PCAResult) -> None:
    """Print a formatted PCA results table to stdout.

    Args:
        result: PCAResult from ``compute_pca``.

    """
    evr = result.explained_variance_ratio
    names = result.variable_names
    importance = result.feature_importance

    print(  # noqa: T201
        f"\nPCA Feature Importance Analysis (samples={result.n_samples}, features={result.n_features})"
    )
    print("-" * 70)  # noqa: T201

    # Explained variance summary.
    print("\nExplained Variance:")  # noqa: T201
    for i, (ratio, cum) in enumerate(
        zip(evr, result.cumulative_explained_variance, strict=True)
    ):
        if cum >= VARIANCE_THRESHOLD and i == np.searchsorted(
            result.cumulative_explained_variance,
            VARIANCE_THRESHOLD,
        ):
            print(  # noqa: T201
                f"  PC{i + 1}: {ratio * 100:5.1f}% (cumulative: {cum * 100:5.1f}%)   <-- 90% threshold reached"
            )
        else:
            print(f"  PC{i + 1}: {ratio * 100:5.1f}% (cumulative: {cum * 100:5.1f}%)")  # noqa: T201

    n_components_90 = (
        int(np.searchsorted(result.cumulative_explained_variance, VARIANCE_THRESHOLD))
        + 1
    )
    print(f"\nComponents needed for 90% variance: {n_components_90}")  # noqa: T201

    # Feature importance ranking.
    print("\nFeature Importance (weighted absolute loadings):")  # noqa: T201
    print("-" * 70)  # noqa: T201
    order = np.argsort(-importance)
    print(f"{'Rank':<5} {'Variable':<45} {'Score':>8}")  # noqa: T201
    print("-" * 70)  # noqa: T201

    for rank, idx in enumerate(order, start=1):
        print(  # noqa: T201
            f"{rank:<5} {names[idx]:<45} {importance[idx]:>8.4f}"
        )

    print()  # noqa: T201


def save_pca_results(result: PCAResult, output_dir: Path) -> Path:
    """Save PCA results to JSON and a text report in the given directory.

    Args:
        result: PCAResult from ``compute_pca``.
        output_dir: Directory to write files into (created if missing).

    Returns:
        Path to the written JSON file.

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report — machine-readable, includes all metadata.
    json_path = output_dir / "pca_results.json"
    serialisable = {
        "variable_names": result.variable_names,
        "explained_variance_ratio": result.explained_variance_ratio.tolist(),
        "cumulative_explained_variance": result.cumulative_explained_variance.tolist(),
        "components": result.components.tolist(),
        "feature_importance": result.feature_importance.tolist(),
        "n_samples": result.n_samples,
        "n_features": result.n_features,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)

    # Text report — human-readable table.
    txt_path = output_dir / "pca_report.txt"
    lines: list[str] = []
    lines.append(
        f"PCA Feature Importance Analysis (samples={result.n_samples}, features={result.n_features})"
    )
    lines.append("-" * 70)

    evr = result.explained_variance_ratio
    cum = result.cumulative_explained_variance
    names = result.variable_names
    importance = result.feature_importance

    lines.append("\nExplained Variance:")
    for i, (ratio, c) in enumerate(zip(evr, cum, strict=True)):
        if c >= VARIANCE_THRESHOLD and i == np.searchsorted(cum, VARIANCE_THRESHOLD):
            lines.append(
                f"  PC{i + 1}: {ratio * 100:5.1f}% (cumulative: {c * 100:5.1f}%)   <-- 90% threshold reached"
            )
        else:
            lines.append(
                f"  PC{i + 1}: {ratio * 100:5.1f}% (cumulative: {c * 100:5.1f}%)"
            )

    n_components_90 = int(np.searchsorted(cum, VARIANCE_THRESHOLD)) + 1
    lines.append(f"\nComponents needed for 90% variance: {n_components_90}")

    lines.append("\nFeature Importance (weighted absolute loadings):")
    lines.append("-" * 70)
    order = np.argsort(-importance)
    lines.append(f"{'Rank':<5} {'Variable':<45} {'Score':>8}")
    lines.append("-" * 70)

    for rank, idx in enumerate(order, start=1):
        lines.append(f"{rank:<5} {names[idx]:<45} {importance[idx]:>8.4f}")

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("PCA results written to %s and %s.", json_path, txt_path)
    return json_path
