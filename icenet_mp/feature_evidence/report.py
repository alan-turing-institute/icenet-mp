"""Consolidate diagnostic and sampled-map RF evidence by input parameter."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from .registry import FeatureRegistry


_CSV_HEADER_AND_DATA_ROWS = 2
_MINIMUM_STABLE_RANK = 0.5


@dataclass(frozen=True)
class EvidenceRow:
    """One parameter's evidence at a forecast lead and spatial stratum."""

    identifier: str
    family: str
    lead: int | None
    stratum: str | None
    vif: float | None
    pca_importance: float | None
    eof_importance: float | None
    rf_mse_improvement: float | None
    rf_group_importance: float | None
    rf_importance_interpretable: bool | None
    rf_importance_calculated: bool | None = None
    target_mode: str | None = None
    max_abs_correlation: float | None = None
    rf_importance_std: float | None = None
    rf_fold_values: list[float] | None = None
    rf_positive_folds: int | None = None
    rf_positive_fold_fraction: float | None = None
    rf_rank_stability: float | None = None
    rf_reliability: str | None = None
    rf_baseline_mse: float | None = None
    rf_mse: float | None = None
    rf_drop_group_mse: float | None = None
    rf_add_group_mse: float | None = None
    rf_add_to_sic_gain: float | None = None
    rf_drop_from_full_loss: float | None = None
    recommendation: str = "inconclusive"


@dataclass(frozen=True)
class ModelQualification:
    """RF skill against persistence for one lead and spatial stratum."""

    lead: int
    stratum: str
    n_samples: int
    rf_mse: float
    persistence_mse: float
    mse_improvement: float
    importance_interpretable: bool


@dataclass(frozen=True)
class EvidenceReport:
    """Separated model qualification, variable evidence, and run provenance."""

    variable_evidence: list[EvidenceRow]
    model_qualification: list[ModelQualification]
    provenance: dict[str, Any]


def _load_json(path: Path | None) -> Mapping[str, Any]:
    """Load an optional JSON analysis result, returning an empty mapping when absent."""
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _named_values(result: Mapping[str, Any], value_key: str) -> dict[str, float]:
    """Pair a result's canonical variable names with one numeric result vector."""
    names = result.get("variable_names", [])
    values = result.get(value_key, [])
    return {str(name): float(value) for name, value in zip(names, values, strict=True)}


def _diagnostic_status(result: Mapping[str, Any], flag: str) -> dict[str, Any] | None:
    """Describe an available diagnostic's evidence qualification."""
    if not result:
        return None
    qualification = result.get("qualification") or {}
    value = result.get(flag) if flag in result else None
    if value is False:
        reason = qualification.get("exclusion_reason")
        if reason is None:
            reason = "Result-level evidence qualification is false."
    elif value is True:
        reason = "Result-level evidence qualification is true."
    else:
        reason = "Qualification was not requested for this run; numeric values are retained."
    return {
        "qualification_flag": flag,
        "qualified": value,
        "reason": reason,
        "qualification": qualification,
    }


def _maximum_absolute_correlations(result: Mapping[str, Any]) -> dict[str, float]:
    """Return each variable's strongest pairwise absolute correlation."""
    names = [str(name) for name in result.get("variable_names", [])]
    matrix = result.get("matrix", [])
    if len(names) != len(matrix):
        return {}
    return {
        name: max(
            (
                abs(float(value))
                for index, value in enumerate(matrix[row])
                if index != row
            ),
            default=0.0,
        )
        for row, name in enumerate(names)
    }


def _load_maximum_absolute_correlations(path: Path | None) -> dict[str, float]:
    """Load correlation evidence from its CSV output or an optional JSON matrix."""
    if path is None or not path.exists():
        return {}
    if path.suffix == ".json":
        return _maximum_absolute_correlations(_load_json(path))
    with path.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.reader(file_handle))
    if len(rows) < _CSV_HEADER_AND_DATA_ROWS:
        return {}
    names = rows[0][1:]
    return {
        row[0]: max(
            (
                abs(float(value))
                for index, value in enumerate(row[1:])
                if names[index] != row[0]
            ),
            default=0.0,
        )
        for row in rows[1:]
        if row
    }


def _recommendation(
    importance: Mapping[str, Any],
    *,
    importance_calculated: bool,
) -> str:
    """Produce an explicitly non-causal screening recommendation."""
    if not importance_calculated:
        return "inconclusive"
    score = importance.get("mean_mse_increase")
    stability = importance.get("rank_stability")
    if score is None:
        return "inconclusive"
    if float(score) <= 0:
        return "deprioritise"
    if stability is not None and float(stability) < _MINIMUM_STABLE_RANK:
        return "investigate"
    return "retain"


def build_evidence_rows(
    registry: FeatureRegistry,
    *,
    vif_path: Path | None = None,
    pca_path: Path | None = None,
    eof_path: Path | None = None,
    correlation_path: Path | None = None,
    spatial_rf_path: Path | None = None,
) -> EvidenceReport:
    """Join registry members with available diagnostic and per-lead RF evidence."""
    vif_result = _load_json(vif_path)
    eof_result = _load_json(eof_path)
    vif = (
        {}
        if vif_result.get("evidence_qualified") is False
        else _named_values(vif_result, "vif_scores")
    )
    pca = _named_values(_load_json(pca_path), "feature_importance")
    eof = (
        {}
        if eof_result.get("independent_feature_selection_evidence") is False
        else _named_values(eof_result, "feature_importance")
    )
    correlations = _load_maximum_absolute_correlations(correlation_path)
    spatial = _load_json(spatial_rf_path)
    leads = spatial.get("leads", [])
    diagnostics = {
        name: status
        for name, status in {
            "vif": _diagnostic_status(vif_result, "evidence_qualified"),
            "eof": _diagnostic_status(
                eof_result, "independent_feature_selection_evidence"
            ),
        }.items()
        if status is not None
    }

    if not leads:
        return EvidenceReport(
            variable_evidence=[
                EvidenceRow(
                    identifier=group.identifier,
                    family=group.family,
                    lead=None,
                    stratum=None,
                    vif=vif.get(group.identifier),
                    pca_importance=pca.get(group.identifier),
                    eof_importance=eof.get(group.identifier),
                    max_abs_correlation=correlations.get(group.identifier),
                    rf_mse_improvement=None,
                    rf_group_importance=None,
                    rf_importance_interpretable=None,
                    rf_importance_calculated=None,
                )
                for group in registry.groups.values()
            ],
            model_qualification=[],
            provenance={"diagnostics": diagnostics} if diagnostics else {},
        )

    rows: list[EvidenceRow] = []
    target_mode = str(spatial.get("metadata", {}).get("target_mode", "absolute"))
    for group in registry.groups.values():
        for lead in leads:
            importance = (lead.get("importance") or {}).get(group.identifier, {})
            interpretable = bool(lead.get("importance_interpretable", False))
            importance_calculated = bool(importance)
            summaries: list[tuple[str | None, Mapping[str, Any]]] = [(None, importance)]
            summaries.extend(
                (str(stratum), summary)
                for stratum, summary in importance.get("by_stratum", {}).items()
            )
            for stratum, summary in summaries:
                rows.append(
                    EvidenceRow(
                        identifier=group.identifier,
                        family=group.family,
                        lead=int(lead["lead"]),
                        stratum=stratum,
                        vif=vif.get(group.identifier),
                        pca_importance=pca.get(group.identifier),
                        eof_importance=eof.get(group.identifier),
                        max_abs_correlation=correlations.get(group.identifier),
                        rf_mse_improvement=None,
                        rf_group_importance=(
                            float(summary["mean_mse_increase"])
                            if "mean_mse_increase" in summary
                            else None
                        ),
                        rf_importance_interpretable=interpretable,
                        rf_importance_calculated=importance_calculated,
                        target_mode=target_mode,
                        rf_importance_std=(
                            float(summary["std_mse_increase"])
                            if "std_mse_increase" in summary
                            else None
                        ),
                        rf_fold_values=(
                            [float(value) for value in summary["fold_values"]]
                            if "fold_values" in summary
                            else None
                        ),
                        rf_positive_folds=(
                            int(summary["positive_folds"])
                            if "positive_folds" in summary
                            else None
                        ),
                        rf_positive_fold_fraction=(
                            float(summary["positive_fold_fraction"])
                            if "positive_fold_fraction" in summary
                            else None
                        ),
                        rf_rank_stability=(
                            float(summary["rank_stability"])
                            if "rank_stability" in summary
                            else None
                        ),
                        rf_reliability=summary.get("reliability"),
                        rf_baseline_mse=None,
                        rf_mse=None,
                        rf_drop_group_mse=(
                            float(importance["drop_group_mse"])
                            if importance.get("drop_group_mse") is not None
                            else None
                        ),
                        rf_add_group_mse=(
                            float(importance["add_group_mse"])
                            if importance.get("add_group_mse") is not None
                            else None
                        ),
                        rf_add_to_sic_gain=(
                            float(importance["add_to_sic_gain"])
                            if importance.get("add_to_sic_gain") is not None
                            else None
                        ),
                        rf_drop_from_full_loss=(
                            float(importance["drop_from_full_loss"])
                            if importance.get("drop_from_full_loss") is not None
                            else None
                        ),
                        recommendation=_recommendation(
                            summary,
                            importance_calculated=importance_calculated,
                        ),
                    )
                )
    qualifications: list[ModelQualification] = []
    for lead in leads:
        interpretable = bool(lead.get("importance_interpretable", False))
        qualification_scores = {
            "all": {
                "n_samples": lead["n_samples"],
                "mse": lead["mse"],
                "baseline_mse": lead["baseline_mse"],
                "mse_improvement": float(lead["baseline_mse"]) - float(lead["mse"]),
            },
            **lead.get("by_stratum", {}),
        }
        for stratum, scores in qualification_scores.items():
            qualifications.append(
                ModelQualification(
                    lead=int(lead["lead"]),
                    stratum=str(stratum),
                    n_samples=int(scores["n_samples"]),
                    rf_mse=float(scores["mse"]),
                    persistence_mse=float(scores["baseline_mse"]),
                    mse_improvement=float(scores["mse_improvement"]),
                    importance_interpretable=interpretable,
                )
            )
    metadata = spatial.get("metadata", {})
    return EvidenceReport(
        variable_evidence=rows,
        model_qualification=qualifications,
        provenance={
            "effective_samples": metadata.get("effective_samples"),
            "feature_count": len(spatial.get("feature_names", [])),
            "initialisation_count": metadata.get("initialisation_count"),
            "target_mode": metadata.get("target_mode", "absolute"),
            "rf_settings": metadata.get("rf_settings", {}),
            **({"diagnostics": diagnostics} if diagnostics else {}),
        },
    )


def save_evidence_report(
    report: EvidenceReport, output_dir: Path
) -> tuple[Path, Path, Path]:
    """Write joined evidence as JSON, CSV and a qualified Markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) for row in report.variable_evidence]
    json_path = output_dir / "parameter_evidence.json"
    warning = (
        "Exploratory screening only: the same held-out folds are used for model "
        "qualification and permutation importance; this is not confirmatory evidence."
    )
    payload = {
        "warning": warning,
        "provenance": report.provenance,
        "model_qualification": [asdict(row) for row in report.model_qualification],
        "variable_evidence": records,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output_dir / "parameter_evidence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=list(records[0])
            if records
            else list(EvidenceRow.__annotations__),
        )
        writer.writeheader()
        writer.writerows(records)

    markdown_path = output_dir / "parameter_evidence.md"
    lines = [
        "# Parameter evidence report",
        "",
        f"**Warning:** {warning}",
        "Importance is predictive and model-specific, not causal. Correlated parameters can be substitutable.",
        "RF-versus-persistence performance is model-quality context, not a gate on screening importance when the opt-in `always` policy is used.",
        "",
        "## Run provenance",
        "",
        f"- Effective samples: {report.provenance.get('effective_samples', '—')}",
        f"- Feature count: {report.provenance.get('feature_count', '—')}",
        f"- Initialisation count: {report.provenance.get('initialisation_count', '—')}",
        f"- Target mode: {report.provenance.get('target_mode', '—')}",
        f"- RF settings: `{json.dumps(report.provenance.get('rf_settings', {}), sort_keys=True)}`",
    ]
    for name, status in report.provenance.get("diagnostics", {}).items():
        qualified = status.get("qualified")
        display = "not reported" if qualified is None else str(qualified).lower()
        lines.append(
            f"- {str(name).upper()} evidence qualified: {display}; reason: {status['reason']}"
        )
    lines.extend(
        [
            "",
            "## Model qualification (RF versus persistence)",
            "",
            "| Lead | Stratum | Effective samples | RF MSE | Persistence MSE | MSE improvement | RF beat persistence |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    lines.extend(
        f"| {qualification.lead} | {qualification.stratum} | {qualification.n_samples} "
        f"| {qualification.rf_mse:.6f} | {qualification.persistence_mse:.6f} "
        f"| {qualification.mse_improvement:.6f} | "
        f"{'yes' if qualification.importance_interpretable else 'no'} |"
        for qualification in report.model_qualification
    )
    lines.extend(
        [
            "",
            "## Variable-level evidence",
            "",
            "Model MSE improvement is qualification evidence and is not a variable contribution.",
            "",
            "| Parameter | Target | Lead/stratum | Permutation importance | Positive folds | Rank stability | Add-to-SIC gain | Drop-from-full loss | Reliability | Recommendation |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in report.variable_evidence:
        importance = (
            f"{row.rf_group_importance:.6f}"
            if row.rf_group_importance is not None
            else "—"
        )
        lead = str(row.lead) if row.lead is not None else "—"
        lead_stratum = f"{lead}/{row.stratum or 'all'}"
        positive_folds = (
            str(row.rf_positive_folds) if row.rf_positive_folds is not None else "—"
        )
        stability = (
            f"{row.rf_rank_stability:.3f}" if row.rf_rank_stability is not None else "—"
        )
        add_gain = (
            f"{row.rf_add_to_sic_gain:.6f}"
            if row.rf_add_to_sic_gain is not None
            else "—"
        )
        drop_loss = (
            f"{row.rf_drop_from_full_loss:.6f}"
            if row.rf_drop_from_full_loss is not None
            else "—"
        )
        lines.append(
            f"| {row.identifier} | {row.target_mode or '—'} | {lead_stratum} | "
            f"{importance} | {positive_folds} | {stability} | {add_gain} | {drop_loss} | "
            f"{row.rf_reliability or '—'} | {row.recommendation} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path
