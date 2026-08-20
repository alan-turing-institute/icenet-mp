"""Shared result container and plotting helpers for input explainability methods.

**Adding a new method:**
1. Create ``icenet_mp/input_explainability/my_method.py`` with a ``compute_my_method()``
   function that returns an :class:`ExplainabilityResult`.
2. Register it in the orchestrator (if you want it run via the unified command).

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003

import numpy as np


@dataclass(frozen=True)
class ExplainabilityResult:
    """Container for explainability analysis results.

    Attributes:
        feature_names: Names of input features used in the model.
        target_name: Name of the predicted target variable (e.g., "sic-ssmis/ice_conc").
        n_samples: Number of samples used for training / evaluation.
        n_features: Number of input features.
        permutation_importance: Mean permutation importance per feature.
        importances_std: Standard deviation of permutation importance across CV folds.
        r2_score: Coefficient of determination (model fit quality).
        mse: Mean squared error on the held-out test set.
        interaction_scores: Optional pairwise interaction scores, shape (n_features, n_features).
            ``interaction_scores[i, j]`` is the interaction strength between features i and j.

    """

    feature_names: list[str]
    target_name: str
    n_samples: int
    n_features: int
    permutation_importance: np.ndarray  # (n_features,)
    importances_std: np.ndarray  # (n_features,) — std across CV folds
    r2_score: float
    mse: float
    interaction_scores: np.ndarray | None = None  # (n_features, n_features) or None


def plot_feature_importance(
    result: ExplainabilityResult,
    output_dir: Path,
    *,
    top_n: int = 30,
) -> Path:
    """Plot feature importance as a horizontal bar chart.

    Args:
        result: Result from an explainability method.
        output_dir: Directory to write the figure into (created if missing).
        top_n: Number of features to display (by absolute importance).

    Returns:
        Path to the saved PNG file.

    """
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)

    importance = result.permutation_importance
    std = result.importances_std
    names = list(result.feature_names)

    # Sort by absolute importance, take top N.
    order = np.argsort(-np.abs(importance))[:top_n]
    sorted_names = [names[i] for i in order]
    sorted_imp = importance[order]
    sorted_std = std[order]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_names) * 0.35)))
    y_pos = np.arange(len(sorted_names))

    bars = ax.barh(
        y_pos,
        sorted_imp,
        xerr=sorted_std,
        align="center",
        color="#2196F3",
        capsize=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (MSE increase)")
    ax.set_title(
        f"Feature Importance — Predicting {result.target_name}\n"
        f"R²={result.r2_score:.3f}, MSE={result.mse:.4f}"
    )

    # Add value labels on bars.
    for bar, val in zip(bars, sorted_imp, strict=True):
        ax.text(
            bar.get_width() + (0.001 if bar.get_width() >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=7,
        )

    fig.tight_layout()
    path = output_dir / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_interaction_heatmap(
    result: ExplainabilityResult,
    output_dir: Path,
    *,
    top_n: int = 20,
) -> Path | None:
    """Plot pairwise feature interaction strengths as a heatmap.

    Uses permutation-based interaction scores: for each pair of features (i, j),
    the score measures how much more model performance degrades when both are shuffled
    together versus the sum of shuffling them individually. Positive values indicate
    synergistic interactions; negative values indicate redundancy.

    Args:
        result: Result from an explainability method. Must have ``interaction_scores``.
        output_dir: Directory to write the figure into (created if missing).
        top_n: Number of features to include (by absolute importance).

    Returns:
        Path to the saved PNG file, or None if no interaction scores are available.

    """
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if result.interaction_scores is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    importance = result.permutation_importance
    names = list(result.feature_names)

    # Select top-N features by absolute importance.
    order = np.argsort(-np.abs(importance))[:top_n]
    sub_idx = list(order)
    sub_names = [names[i] for i in sub_idx]
    n = len(sub_idx)

    interactions = result.interaction_scores[np.ix_(sub_idx, sub_idx)]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(6, n * 0.4)))
    im = ax.imshow(
        interactions, cmap="RdBu_r", vmin=-interactions.max(), vmax=interactions.max()
    )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(sub_names, fontsize=7)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    ax.set_title(f"Pairwise Feature Interactions — Predicting {result.target_name}")

    # Add value annotations.
    for i in range(n):
        for j in range(n):
            val = interactions[i, j]
            color = "white" if abs(val) > interactions.max() * 0.5 else "black"
            ax.text(
                j, i, f"{val:.3f}", ha="center", va="center", fontsize=6, color=color
            )

    fig.colorbar(im, ax=ax, shrink=0.8, label="Interaction Strength")
    fig.tight_layout()

    path = output_dir / "interaction_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path
