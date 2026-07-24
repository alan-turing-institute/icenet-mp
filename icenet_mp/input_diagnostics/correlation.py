r"""Correlation heatmap analysis for input variables.

Computes Pearson correlation matrices and generates visual heatmaps to complement
VIF, PCA, and EOF analyses. This helps identify pairs of variables that are highly
correlated (which may indicate redundancy).

Usage::

    from icenet_mp.input_diagnostics.correlation import (
        compute_correlation_matrix,
        plot_correlation_heatmap,
        print_correlation_summary,
        save_correlation_csv,
    )

"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np  # noqa: TC002 — used at runtime in print_correlation_summary
import pandas as pd

logger = logging.getLogger(__name__)


def compute_correlation_matrix(
    sample_matrix: np.ndarray,
    var_names: list[str],
) -> pd.DataFrame:
    """Compute Pearson correlation matrix from a sample matrix.

    Args:
        sample_matrix: Feature matrix of shape (n_samples, n_features).
        var_names: Variable names corresponding to columns.

    Returns:
        Pandas DataFrame with variable names as index and columns.

    """
    df = pd.DataFrame(sample_matrix, columns=list(var_names))
    return df.corr(method="pearson")


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    output_path: Path | str,
) -> None:
    """Plot a correlation heatmap with clustermap.

    Args:
        corr_df: Correlation DataFrame from ``compute_correlation_matrix``.
        output_path: Path to save the figure (PNG).

    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_df.index, fontsize=8)
    ax.set_title("Pearson Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Correlation heatmap written to %s", output_path)


def print_correlation_summary(corr_df: pd.DataFrame, top_n: int = 10) -> None:
    """Print a summary of the strongest correlations.

    Args:
        corr_df: Correlation DataFrame from ``compute_correlation_matrix``.
        top_n: Number of top correlations to display.

    """
    pairs = [
        (str(corr_df.columns[i]), str(corr_df.columns[j]), float(corr_df.iloc[i, j]))  # type: ignore[arg-type]
        for i in range(len(corr_df))
        for j in range(i + 1, len(corr_df))
    ]

    # Sort by absolute correlation.
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\nTop {top_n} Correlations:")  # noqa: T201 — intentional CLI output
    print("-" * 70)  # noqa: T201
    print(f"{'Pair':<50} {'Correlation':>12}")  # noqa: T201
    print("-" * 70)  # noqa: T201
    for var1, var2, corr in pairs[:top_n]:
        print(f"{var1} ↔ {var2:<30} {corr:12.4f}")  # noqa: T201
    print()  # noqa: T201


def save_correlation_csv(corr_df: pd.DataFrame, output_dir: Path | str) -> Path:
    """Save correlation matrix to CSV.

    Args:
        corr_df: Correlation DataFrame from ``compute_correlation_matrix``.
        output_dir: Directory to save the CSV file.

    Returns:
        Path to the saved CSV file.

    """
    out_path = Path(output_dir) / "correlations.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_path)
    logger.info("Correlation matrix saved to %s", out_path)
    return out_path
