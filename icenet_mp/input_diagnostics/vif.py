"""VIF (Variance Inflation Factor) analysis for input variable multicollinearity.

Computes per-variable VIF scores using statsmodels OLS regression. Variables with
high VIF are redundant — they can be predicted well from other variables in the set,
suggesting one could be removed without losing information.

**Interpretation:**
- VIF = 1: no correlation with other variables.
- VIF > 5 (default threshold): moderately to highly multicollinear — consider removing.
- VIF > 10: severely multicollinear — should almost certainly be removed or combined.

VIF is **unsupervised**: it measures redundancy among input variables alone, not their
predictive power for the target (SIC). A variable can have high VIF and still be useful
for prediction if all its redundant partners are also kept.

"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path  # noqa: TC003

import numpy as np
from omegaconf import DictConfig  # noqa: TC002
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .data import build_datasets, build_sample_matrix, resolve_datasets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VIFResult:
    """Holds the results of a VIF analysis run."""

    # Variable names in column order (group_as/variable pairs).
    variable_names: list[str]
    # Per-variable VIF scores, one per column.
    vif_scores: np.ndarray
    # Threshold used for flagging.
    threshold: float
    # Number of samples (timesteps) used.
    n_samples: int


def compute_vif(
    sample_matrix: np.ndarray,
    variable_names: list[str],
    threshold: float = 5.0,
) -> VIFResult:
    """Compute VIF scores for each column in the sample matrix.

    Args:
        sample_matrix: Array of shape (n_samples, n_variables).
        variable_names: One name per column.
        threshold: VIF value above which a variable is flagged as multicollinear.

    Returns:
        VIFResult with scores and metadata.

    """
    n_vars = sample_matrix.shape[1]
    vif_scores = np.empty(n_vars)

    # Add constant for OLS regression (required by statsmodels VIF).
    X_ = np.column_stack(  # noqa: N806
        [np.ones(sample_matrix.shape[0]), sample_matrix]
    )

    n_samples = sample_matrix.shape[0]
    if n_samples < n_vars + 1:
        msg = (
            f"VIF requires more samples than variables "
            f"(got {n_samples} samples, {n_vars} variables). "
            f"Increase max_samples or reduce the number of input variables."
        )
        raise ValueError(msg)

    for i in range(n_vars):
        vif_scores[i] = variance_inflation_factor(X_, i + 1)  # +1 for constant column

    flagged = [
        name
        for name, score in zip(variable_names, vif_scores, strict=True)
        if score > threshold
    ]

    logger.info(
        "VIF analysis complete: %d variables, %d flagged (threshold=%.1f).",
        n_vars,
        len(flagged),
        threshold,
    )

    return VIFResult(
        variable_names=variable_names,
        vif_scores=vif_scores,
        threshold=threshold,
        n_samples=sample_matrix.shape[0],
    )


def run_vif_analysis(config: DictConfig) -> VIFResult:
    """Run a full VIF analysis from a Hydra-composed config.

    Resolves dataset paths and variable selections directly from the config (without
    requiring a prediction target), loads raw data via SingleDataset, and computes
    per-variable VIF scores.

    Args:
        config: Hydra-composed config (from ``imp vif``).

    Returns:
        VIFResult with computed scores.

    """
    vif_cfg = config.get("vif", {})
    threshold = float(vif_cfg.get("threshold", 5.0))
    max_samples = vif_cfg.get("max_samples", None)

    # Resolve dataset paths and variable selections from config.
    group_paths, group_variables = resolve_datasets(config)

    # Build SingleDataset instances for each group.
    datasets = build_datasets(group_paths, group_variables)

    sample_matrix, var_names = build_sample_matrix(datasets, max_samples=max_samples)

    logger.info("Sample matrix shape: %s (%d variables).", sample_matrix.shape, len(var_names))

    return compute_vif(sample_matrix, var_names, threshold=threshold)


def _format_table_lines(result: VIFResult) -> tuple[list[str], int]:
    """Format VIF results as table lines sorted by score descending.

    Shared helper for ``print_vif_table`` and ``save_vif_results`` to avoid duplication.

    Args:
        result: VIFResult from ``compute_vif`` or ``run_vif_analysis``.

    Returns:
        Tuple of (table_lines, flagged_count).

    """
    threshold = result.threshold
    scores = result.vif_scores
    names = result.variable_names

    lines: list[str] = []
    order = np.argsort(-scores)

    for idx in order:
        name = names[idx]
        score = scores[idx]
        flag = " ***" if score > threshold else ""
        lines.append(f"{name:<45} {score:>8.2f}{flag}")

    flagged = sum(1 for s in scores if s > threshold)
    return lines, flagged


def print_vif_table(result: VIFResult) -> None:
    """Print a formatted VIF results table to stdout.

    Args:
        result: VIFResult from ``compute_vif`` or ``run_vif_analysis``.

    """
    threshold = result.threshold
    names = result.variable_names

    print(  # noqa: T201
        f"\nVIF Analysis Results (threshold={threshold:.1f}, samples={result.n_samples})"
    )
    print("-" * 70)  # noqa: T201
    print(f"{'Variable':<45} {'VIF':>8}")  # noqa: T201
    print("-" * 70)  # noqa: T201

    lines, flagged = _format_table_lines(result)
    for line in lines:
        print(line)  # noqa: T201

    print("-" * 70)  # noqa: T201
    print(  # noqa: T201
        f"Total: {len(names)} variables, {flagged} above threshold ({threshold:.1f})"
    )
    print()  # noqa: T201


def save_vif_results(result: VIFResult, output_dir: Path) -> Path:
    """Save VIF results to JSON and a text report in the given directory.

    Args:
        result: VIFResult from ``compute_vif`` or ``run_vif_analysis``.
        output_dir: Directory to write files into (created if missing).

    Returns:
        Path to the written JSON file.

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report — machine-readable, includes all metadata.
    json_path = output_dir / "vif_results.json"
    serialisable = asdict(result)
    serialisable["vif_scores"] = result.vif_scores.tolist()  # ndarray → list for JSON
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)

    # Text report — human-readable table.
    txt_path = output_dir / "vif_report.txt"
    lines: list[str] = []
    lines.append(f"VIF Analysis Results (threshold={result.threshold:.1f}, samples={result.n_samples})")
    lines.append("-" * 70)
    lines.append(f"{'Variable':<45} {'VIF':>8}")
    lines.append("-" * 70)

    table_lines, flagged = _format_table_lines(result)
    lines.extend(table_lines)
    lines.append("-" * 70)
    lines.append(f"Total: {len(result.variable_names)} variables, {flagged} above threshold ({result.threshold:.1f})")

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("VIF results written to %s and %s.", json_path, txt_path)
    return json_path
