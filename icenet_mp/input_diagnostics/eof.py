"""EOF (Empirical Orthogonal Function) analysis for input variable structure.

Centers the data and performs SVD to decompose variance into orthogonal modes.
In geoscience, EOFs represent dominant spatiotemporal patterns of variability —
the first mode captures the largest fraction of total variance, the second captures
the largest remaining fraction orthogonal to the first, and so on.

**Interpretation:**
- Each **EOF mode** (component) is a direction in variable space that explains
  a portion of total input variance.
- A variable with high loading on an EOF mode co-varies strongly with other variables
  in that mode — they are part of the same physical/statistical pattern.
- The explained variance fraction tells you how important each mode is overall.

EOF is mathematically equivalent to PCA (both use SVD on the covariance structure) but
uses geoscience terminology and conventions. Use it when you want to interpret modes as
physical patterns (e.g., "this EOF represents the NAO pressure dipole").

Like all diagnostics, EOF is **unsupervised** — it does not use the target (SIC).
High-loading variables are those that vary together in structured ways across timesteps.

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import numpy as np

from .data import build_sample_matrix, resolve_datasets

if TYPE_CHECKING:
    from omegaconf import DictConfig


logger = logging.getLogger(__name__)

# Minimum dimensions for SVD to be meaningful.
_MIN_ROWS = 2   # at least 2 timesteps
_MIN_COLS = 2   # at least 2 variables

# Threshold for cumulative explained variance (modes needed to capture 90%).
_VARIANCE_THRESHOLD = 0.90


@dataclass(frozen=True)
class EOFResult:
    """Holds the results of an EOF analysis run."""

    variable_names: list[str]
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    eof_modes: np.ndarray  # (n_modes, n_features) — each row is a mode's loadings
    feature_importance: np.ndarray  # variance-weighted absolute loadings per variable
    n_samples: int
    n_features: int


def compute_eof(
    sample_matrix: np.ndarray,
    variable_names: list[str],
) -> EOFResult:
    """Compute EOF analysis via SVD and derive per-variable importance.

    Args:
        sample_matrix: Array of shape ``(n_samples, n_variables)`` — already spatially
            aggregated, one row per timestep.
        variable_names: One name per column.

    Returns:
        EOFResult with modes, explained variance, and metadata.

    Raises:
        ValueError: If the matrix has fewer than 2 rows or columns (SVD undefined).

    """
    n_rows, n_cols = sample_matrix.shape

    if n_rows < _MIN_ROWS or n_cols < _MIN_COLS:
        msg = (
            f"EOF requires at least {_MIN_ROWS} timesteps and {_MIN_COLS} variables "
            f"(got {n_rows} timesteps, {n_cols} variables)."
        )
        raise ValueError(msg)

    # Center the data (subtract mean of each variable).
    centered = sample_matrix - sample_matrix.mean(axis=0)

    # SVD: U @ diag(s) @ Vt.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)

    # Explained variance: proportional to squared singular values.
    total_var = np.sum(s ** 2)
    explained_variance_ratio = (s ** 2) / total_var
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # EOF modes are rows of Vt — each row is a loading vector across variables.
    eof_modes = vt  # shape: (n_modes, n_features)

    # Feature importance: variance-weighted absolute loadings.
    loadings = np.abs(eof_modes.T)  # (n_features, n_modes)
    feature_importance = loadings @ explained_variance_ratio  # (n_features,)

    logger.info(
        "EOF complete: %d samples, %d features, top mode explains %.1f%% variance.",
        sample_matrix.shape[0],
        len(variable_names),
        explained_variance_ratio[0] * 100,
    )

    return EOFResult(
        variable_names=variable_names,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_explained_variance=cumulative_explained_variance,
        eof_modes=eof_modes,
        feature_importance=feature_importance,
        n_samples=sample_matrix.shape[0],
        n_features=len(variable_names),
    )


def run_eof_analysis(config: DictConfig) -> EOFResult:
    """Run a full EOF analysis from a Hydra-composed config.

    Resolves dataset paths and variable selections directly from the config, loads raw
    data via SingleDataset, spatially aggregates, and computes per-variable importance.

    Args:
        config: Hydra-composed config (from ``imp eof``).

    Returns:
        EOFResult with computed modes.

    """
    max_samples = config.get("vif", {}).get("max_samples", None)  # reuse vif.max_samples key

    group_paths, group_variables = resolve_datasets(config)

    from icenet_mp.data_loaders.single_dataset import SingleDataset  # noqa: PLC0415

    datasets: dict[str, SingleDataset] = {}
    for group_name, paths in group_paths.items():
        vars_list = group_variables.get(group_name)
        if vars_list is not None and len(vars_list) > 0:
            logger.info("Loading dataset group %r (%d paths, variables: %s).",
                         group_name, len(paths), vars_list)
            datasets[group_name] = SingleDataset(
                group_name,
                paths,
                variables=vars_list,
            )
        else:
            logger.warning(
                "Dataset group %r has no variable filter — loading all variables (%d paths). "
                "Consider specifying 'variables' to limit data loaded.",
                group_name, len(paths),
            )
            datasets[group_name] = SingleDataset(group_name, paths)

    sample_matrix, var_names = build_sample_matrix(datasets, max_samples=max_samples)

    logger.info("Sample matrix shape: %s (%d variables).", sample_matrix.shape, len(var_names))

    return compute_eof(sample_matrix, var_names)


def print_eof_table(result: EOFResult) -> None:
    """Print a formatted EOF results table to stdout.

    Args:
        result: EOFResult from ``compute_eof`` or ``run_eof_analysis``.

    """
    evr = result.explained_variance_ratio
    names = result.variable_names
    importance = result.feature_importance

    print(  # noqa: T201
        f"\nEOF Mode Analysis (samples={result.n_samples}, features={result.n_features})"
    )
    print("-" * 70)  # noqa: T201

    # Explained variance summary.
    print("\nExplained Variance:")  # noqa: T201
    for i, (ratio, cum) in enumerate(zip(evr, result.cumulative_explained_variance, strict=True)):
        if cum >= _VARIANCE_THRESHOLD and i == np.searchsorted(
            result.cumulative_explained_variance, _VARIANCE_THRESHOLD,
        ):
            print(f"  Mode {i + 1}: {ratio * 100:5.1f}% (cumulative: {cum * 100:5.1f}%)   <-- 90% threshold reached")  # noqa: T201
        else:
            print(f"  Mode {i + 1}: {ratio * 100:5.1f}% (cumulative: {cum * 100:5.1f}%)")  # noqa: T201

    n_modes_90 = int(np.searchsorted(result.cumulative_explained_variance, _VARIANCE_THRESHOLD)) + 1
    print(f"\nModes needed for 90% variance: {n_modes_90}")  # noqa: T201

    # Feature importance ranking.
    print("\nFeature Importance (variance-weighted absolute loadings):")  # noqa: T201
    print("-" * 70)  # noqa: T201
    order = np.argsort(-importance)
    print(f"{'Rank':<5} {'Variable':<45} {'Score':>8}")  # noqa: T201
    print("-" * 70)  # noqa: T201

    for rank, idx in enumerate(order, start=1):
        print(  # noqa: T201
            f"{rank:<5} {names[idx]:<45} {importance[idx]:>8.4f}"
        )

    print()  # noqa: T201


def save_eof_results(result: EOFResult, output_dir: Path) -> Path:
    """Save EOF results to JSON and a text report in the given directory.

    Args:
        result: EOFResult from ``compute_eof`` or ``run_eof_analysis``.
        output_dir: Directory to write files into (created if missing).

    Returns:
        Path to the written JSON file.

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report — machine-readable, includes all metadata.
    json_path = output_dir / "eof_results.json"
    serialisable = {
        "variable_names": result.variable_names,
        "explained_variance_ratio": result.explained_variance_ratio.tolist(),
        "cumulative_explained_variance": result.cumulative_explained_variance.tolist(),
        "eof_modes": result.eof_modes.tolist(),
        "feature_importance": result.feature_importance.tolist(),
        "n_samples": result.n_samples,
        "n_features": result.n_features,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)

    # Text report — human-readable table.
    txt_path = output_dir / "eof_report.txt"
    lines: list[str] = []
    lines.append(
        f"EOF Mode Analysis (samples={result.n_samples}, features={result.n_features})"
    )
    lines.append("-" * 70)

    evr = result.explained_variance_ratio
    cum = result.cumulative_explained_variance
    names = result.variable_names
    importance = result.feature_importance

    lines.append("\nExplained Variance:")
    for i, (ratio, c) in enumerate(zip(evr, cum, strict=True)):
        if c >= _VARIANCE_THRESHOLD and i == np.searchsorted(cum, _VARIANCE_THRESHOLD):
            lines.append(f"  Mode {i + 1}: {ratio * 100:5.1f}% (cumulative: {c * 100:5.1f}%)   <-- 90% threshold reached")
        else:
            lines.append(f"  Mode {i + 1}: {ratio * 100:5.1f}% (cumulative: {c * 100:5.1f}%)")

    n_modes_90 = int(np.searchsorted(cum, _VARIANCE_THRESHOLD)) + 1
    lines.append(f"\nModes needed for 90% variance: {n_modes_90}")

    lines.append("\nFeature Importance (variance-weighted absolute loadings):")
    lines.append("-" * 70)
    order = np.argsort(-importance)
    lines.append(f"{'Rank':<5} {'Variable':<45} {'Score':>8}")
    lines.append("-" * 70)

    for rank, idx in enumerate(order, start=1):
        lines.append(
            f"{rank:<5} {names[idx]:<45} {importance[idx]:>8.4f}"
        )

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("EOF results written to %s and %s.", json_path, txt_path)
    return json_path
