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
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

_MINIMUM_VARIABLES = 2
_NEAR_SINGULAR_CONDITION = 1.0e12


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
    # Evidence qualification is opt-in; None means qualification wasn't requested,
    # distinct from a computed True/False.
    evidence_qualified: bool | None = None
    qualification: dict[str, object] | None = None


def compute_vif(  # noqa: PLR0913
    sample_matrix: np.ndarray,
    variable_names: list[str],
    threshold: float = 5.0,
    *,
    qualification_enabled: bool = False,
    minimum_sample_feature_ratio: float = 10.0,
    date_range: tuple[str, str] | None = None,
    sampling_limit: int | None = None,
) -> VIFResult:
    """Compute VIF scores for each column in the sample matrix.

    Args:
        sample_matrix: Array of shape (n_samples, n_variables).
        variable_names: One name per column.
        threshold: VIF value above which a variable is flagged as multicollinear.
        qualification_enabled: Whether to attach feature-evidence qualification metadata.
        minimum_sample_feature_ratio: Required observations per predictor for qualification.
        date_range: First and last available common dates, when known.
        sampling_limit: Configured timestep cap, when set.

    Returns:
        VIFResult with scores and metadata.

    Raises:
        ValueError: If fewer than 2 variables are provided (VIF is meaningless for
            a single variable).

    """
    n_vars = sample_matrix.shape[1]

    if n_vars < _MINIMUM_VARIABLES:
        msg = f"VIF requires at least 2 variables to compute multicollinearity; got {n_vars}."
        logger.warning(msg)
        raise ValueError(msg)

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

    qualification: dict[str, object] | None = None
    evidence_qualified: bool | None = None
    if qualification_enabled:
        centered = sample_matrix - sample_matrix.mean(axis=0)
        scales = np.std(centered, axis=0)
        scale_tolerance = np.finfo(float).eps * np.maximum(
            1.0, np.max(np.abs(sample_matrix), axis=0)
        )
        constant_mask = scales <= scale_tolerance
        constant_columns = [
            name
            for name, is_constant in zip(variable_names, constant_mask, strict=True)
            if is_constant
        ]
        standardised = centered[:, ~constant_mask] / scales[~constant_mask]
        ratio = n_samples / n_vars
        rank = int(np.linalg.matrix_rank(standardised)) if standardised.shape[1] else 0
        condition_number = (
            float(np.linalg.cond(standardised)) if standardised.shape[1] else None
        )
        exact_rank_deficient = rank < standardised.shape[1]
        near_rank_deficient = (
            condition_number is None
            or not np.isfinite(condition_number)
            or condition_number >= _NEAR_SINGULAR_CONDITION
        )
        evidence_qualified = (
            ratio >= minimum_sample_feature_ratio
            and not constant_columns
            and not exact_rank_deficient
            and not near_rank_deficient
        )
        qualification = {
            "date_range": list(date_range) if date_range is not None else None,
            "sample_count": n_samples,
            "predictor_count": n_vars,
            "sample_to_feature_ratio": ratio,
            "minimum_sample_feature_ratio": minimum_sample_feature_ratio,
            "diagnostic_predictor_count": standardised.shape[1],
            "constant_columns": constant_columns,
            "matrix_rank": rank,
            "condition_number": condition_number,
            "condition_number_basis": "centred, standardised nonconstant predictors",
            "near_singular_condition_threshold": _NEAR_SINGULAR_CONDITION,
            "exact_rank_deficient": exact_rank_deficient,
            "near_rank_deficient": near_rank_deficient,
            "sampling_limit": sampling_limit,
            "sampling_limitation": (
                "Timesteps are evenly sampled from available common dates when a limit "
                "is set; spatial means discard within-grid variation."
            ),
            "independence_warning": (
                "Temporal autocorrelation reduces the effective number of independent samples."
            ),
            "qualified": evidence_qualified,
        }

    return VIFResult(
        variable_names=variable_names,
        vif_scores=vif_scores,
        threshold=threshold,
        n_samples=sample_matrix.shape[0],
        evidence_qualified=evidence_qualified,
        qualification=qualification,
    )


def _format_table_lines(result: VIFResult) -> tuple[list[str], int]:
    """Format VIF results as table lines sorted by score descending.

    Shared helper for ``print_vif_table`` and ``save_vif_results`` to avoid duplication.

    Args:
        result: VIFResult from ``compute_vif``.

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
        result: VIFResult from ``compute_vif``.

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
        result: VIFResult from ``compute_vif``.
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
    lines.append(
        f"VIF Analysis Results (threshold={result.threshold:.1f}, samples={result.n_samples})"
    )
    lines.append("-" * 70)
    lines.append(f"{'Variable':<45} {'VIF':>8}")
    lines.append("-" * 70)

    table_lines, flagged = _format_table_lines(result)
    lines.extend(table_lines)
    lines.append("-" * 70)
    lines.append(
        f"Total: {len(result.variable_names)} variables, {flagged} above threshold ({result.threshold:.1f})"
    )
    if result.qualification is not None:
        q = result.qualification
        lines.extend(
            [
                "",
                f"Feature-evidence qualification: {'QUALIFIED' if result.evidence_qualified else 'NOT QUALIFIED'}",
                f"Dates: {q['date_range']}; samples/predictors: {q['sample_count']}/{q['predictor_count']}; ratio: {q['sample_to_feature_ratio']:.2f}:1 (minimum {q['minimum_sample_feature_ratio']}:1)",
                f"Constant columns: {q['constant_columns']}",
                f"Standardised nonconstant matrix rank: {q['matrix_rank']}/{q['diagnostic_predictor_count']}; condition number: {q['condition_number']}",
                f"Exact rank deficient: {q['exact_rank_deficient']}; near rank deficient: {q['near_rank_deficient']} (condition threshold {q['near_singular_condition_threshold']})",
                str(q["sampling_limitation"]),
                str(q["independence_warning"]),
            ]
        )

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("VIF results written to %s and %s.", json_path, txt_path)
    return json_path
