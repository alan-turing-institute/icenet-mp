"""Variable-space covariance decomposition (EOF-style mode analysis).

This implementation spatially averages each variable, centers those time series without
standardising them, and performs SVD. Its modes are directions across variables, not
spatial EOF patterns. The first mode captures the largest fraction of variable-space
covariance, followed by orthogonal directions of decreasing variance.

**Interpretation:**
- Each **EOF mode** (component) is a direction in variable space that explains
  a portion of total input variance.
- A variable with high loading on a mode contributes strongly to that variable-space
  covariance direction; this alone does not establish a physical pattern or usefulness.
- The explained variance fraction tells you how important each mode is overall.

The calculation is mathematically equivalent to unstandardised PCA on the spatial means,
so it is excluded as independent feature-selection evidence in the evidence report
(counting both would double-count the same signal).

Like all diagnostics, EOF is **unsupervised** — it does not use the target (SIC).
High-loading variables are those that vary together in structured ways across timesteps.

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003

import numpy as np

logger = logging.getLogger(__name__)

# Minimum dimensions for SVD to be meaningful.
_MIN_ROWS = 2  # at least 2 timesteps
_MIN_COLS = 2  # at least 2 variables

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
    analysis_space: str = "variable-space covariance decomposition over spatial means"
    independent_feature_selection_evidence: bool = False
    qualification: dict[str, object] | None = None


def compute_eof(
    sample_matrix: np.ndarray,
    variable_names: list[str],
    *,
    qualification_enabled: bool = False,
    date_range: tuple[str, str] | None = None,
    sampling_limit: int | None = None,
) -> EOFResult:
    """Decompose variable-space covariance via SVD and derive loading summaries.

    Args:
        sample_matrix: Array of shape ``(n_samples, n_variables)`` — already spatially
            aggregated, one row per timestep.
        variable_names: One name per column.
        qualification_enabled: Whether to attach evidence-suite diagnostic metadata.
        date_range: First and last available common dates, when known.
        sampling_limit: Configured timestep cap, when set.

    Returns:
        EOFResult with variable-space modes, explained variance, and metadata.

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
    total_var = np.sum(s**2)
    explained_variance_ratio = (s**2) / total_var
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

    qualification: dict[str, object] | None = None
    if qualification_enabled:
        qualification = {
            "date_range": list(date_range) if date_range is not None else None,
            "sample_count": n_rows,
            "predictor_count": n_cols,
            "sample_to_feature_ratio": n_rows / n_cols,
            "matrix_rank": int(np.linalg.matrix_rank(centered)),
            "condition_number": float(np.linalg.cond(centered)),
            "sampling_limit": sampling_limit,
            "sampling_limitation": (
                "Timesteps are evenly sampled from available common dates when a limit "
                "is set; spatial means discard within-grid variation."
            ),
            "independence_warning": (
                "Temporal autocorrelation reduces the effective number of independent samples."
            ),
            "qualified": False,
            "exclusion_reason": (
                "Variable-space covariance decomposition is not independent "
                "feature-selection evidence."
            ),
        }

    return EOFResult(
        variable_names=variable_names,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_explained_variance=cumulative_explained_variance,
        eof_modes=eof_modes,
        feature_importance=feature_importance,
        n_samples=sample_matrix.shape[0],
        n_features=len(variable_names),
        qualification=qualification,
    )


def print_eof_table(result: EOFResult) -> None:
    """Print a formatted EOF results table to stdout.

    Args:
        result: EOFResult from ``compute_eof``.

    """
    evr = result.explained_variance_ratio
    names = result.variable_names
    importance = result.feature_importance

    print(  # noqa: T201
        f"\nEOF Variable-Space Covariance Decomposition (samples={result.n_samples}, features={result.n_features})"
    )
    print("-" * 70)  # noqa: T201
    print(  # noqa: T201
        "Spatial means are centered but not standardised; this is not a spatial EOF and is excluded as independent feature-selection evidence."
    )

    # Explained variance summary.
    print("\nExplained Variance:")  # noqa: T201
    for i, (ratio, cum) in enumerate(
        zip(evr, result.cumulative_explained_variance, strict=True)
    ):
        if cum >= _VARIANCE_THRESHOLD and i == np.searchsorted(
            result.cumulative_explained_variance,
            _VARIANCE_THRESHOLD,
        ):
            print(  # noqa: T201
                f"  Mode {i + 1}: {ratio * 100:5.1f}% (cumulative: {cum * 100:5.1f}%)   <-- 90% threshold reached"
            )
        else:
            print(  # noqa: T201
                f"  Mode {i + 1}: {ratio * 100:5.1f}% (cumulative: {cum * 100:5.1f}%)"
            )

    n_modes_90 = (
        int(np.searchsorted(result.cumulative_explained_variance, _VARIANCE_THRESHOLD))
        + 1
    )
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
        result: EOFResult from ``compute_eof``.
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
        "analysis_space": result.analysis_space,
        "standardised": False,
        "independent_feature_selection_evidence": result.independent_feature_selection_evidence,
        "qualification": result.qualification,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)

    # Text report — human-readable table.
    txt_path = output_dir / "eof_report.txt"
    lines: list[str] = []
    lines.append(
        f"EOF Variable-Space Covariance Decomposition (samples={result.n_samples}, features={result.n_features})"
    )
    lines.append("-" * 70)
    lines.append(
        "Spatial means are centered but not standardised; this is not a spatial EOF and is excluded as independent feature-selection evidence."
    )
    if result.qualification is not None:
        q = result.qualification
        lines.append(
            f"Dates: {q['date_range']}; samples/predictors: {q['sample_count']}/{q['predictor_count']}; ratio: {q['sample_to_feature_ratio']:.2f}:1"
        )
        lines.append(
            f"Centered matrix rank: {q['matrix_rank']}; condition number: {q['condition_number']}"
        )
        lines.append(str(q["sampling_limitation"]))
        lines.append(str(q["independence_warning"]))

    evr = result.explained_variance_ratio
    cum = result.cumulative_explained_variance
    names = result.variable_names
    importance = result.feature_importance

    lines.append("\nExplained Variance:")
    for i, (ratio, c) in enumerate(zip(evr, cum, strict=True)):
        if c >= _VARIANCE_THRESHOLD and i == np.searchsorted(cum, _VARIANCE_THRESHOLD):
            lines.append(
                f"  Mode {i + 1}: {ratio * 100:5.1f}% (cumulative: {c * 100:5.1f}%)   <-- 90% threshold reached"
            )
        else:
            lines.append(
                f"  Mode {i + 1}: {ratio * 100:5.1f}% (cumulative: {c * 100:5.1f}%)"
            )

    n_modes_90 = int(np.searchsorted(cum, _VARIANCE_THRESHOLD)) + 1
    lines.append(f"\nModes needed for 90% variance: {n_modes_90}")

    lines.append("\nFeature Importance (variance-weighted absolute loadings):")
    lines.append("-" * 70)
    order = np.argsort(-importance)
    lines.append(f"{'Rank':<5} {'Variable':<45} {'Score':>8}")
    lines.append("-" * 70)

    for rank, idx in enumerate(order, start=1):
        lines.append(f"{rank:<5} {names[idx]:<45} {importance[idx]:>8.4f}")

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("EOF results written to %s and %s.", json_path, txt_path)
    return json_path
