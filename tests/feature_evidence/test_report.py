"""Tests for consolidated parameter-evidence reports."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from icenet_mp.feature_evidence.registry import FeatureRegistry, load_feature_registry
from icenet_mp.feature_evidence.report import build_evidence_rows, save_evidence_report


def _write(path: Path, data: object) -> Path:
    """Write a JSON fixture and return its path."""
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _registry() -> FeatureRegistry:
    """Create a one-variable feature registry fixture."""
    return load_feature_registry(
        OmegaConf.create(
            {
                "feature_evidence": {
                    "registry": {
                        "entries": [
                            {
                                "source": "era5",
                                "variable": "2t",
                                "family": "temperature",
                            }
                        ]
                    }
                }
            }
        )
    )


def _spatial_result(
    *, interpretable: bool, retain_importance: bool | None = None
) -> dict[str, object]:
    """Create a complete sampled-map RF result with optional variable importance."""
    return {
        "feature_names": ["era5/2t_t-0", "era5/2t_t-1"],
        "metadata": {
            "effective_samples": 40,
            "initialisation_count": 10,
            "rf_settings": {"n_estimators": 20, "permutation_repeats": 2},
        },
        "leads": [
            {
                "lead": 1,
                "n_samples": 20,
                "mse": 0.4 if interpretable else 0.6,
                "baseline_mse": 0.5,
                "importance_interpretable": interpretable,
                "importance": (
                    {"era5/2t": {"mean_mse_increase": 0.2}}
                    if retain_importance is True
                    or (retain_importance is None and interpretable)
                    else None
                ),
                "by_stratum": {
                    "marginal_ice": {
                        "n_samples": 8,
                        "mse": 0.4,
                        "baseline_mse": 0.5,
                        "mse_improvement": 0.1,
                    }
                },
            }
        ],
    }


def test_non_null_importance_is_separate_from_model_qualification(
    tmp_path: Path,
) -> None:
    """Variable importance does not inherit whole-model stratum skill."""
    registry = _registry()
    vif_path = _write(
        tmp_path / "vif.json", {"variable_names": ["era5/2t"], "vif_scores": [2.0]}
    )
    spatial_path = _write(
        tmp_path / "spatial.json",
        _spatial_result(interpretable=True),
    )

    report = build_evidence_rows(
        registry, vif_path=vif_path, spatial_rf_path=spatial_path
    )

    evidence = report.variable_evidence[0]
    assert evidence.identifier == "era5/2t"
    assert evidence.vif == 2.0
    assert evidence.rf_group_importance == 0.2
    assert evidence.rf_mse_improvement is None
    assert evidence.stratum is None
    assert {row.stratum for row in report.model_qualification} == {
        "all",
        "marginal_ice",
    }


def test_null_importance_remains_inconclusive_and_is_omitted(tmp_path: Path) -> None:
    """A model that loses to persistence supplies no variable contribution evidence."""
    spatial_path = _write(
        tmp_path / "spatial.json", _spatial_result(interpretable=False)
    )

    report = build_evidence_rows(_registry(), spatial_rf_path=spatial_path)
    evidence = report.variable_evidence[0]

    assert evidence.rf_group_importance is None
    assert evidence.rf_drop_group_mse is None
    assert evidence.rf_add_group_mse is None
    assert evidence.recommendation == "inconclusive"


def test_screening_importance_is_retained_separately_from_model_quality(
    tmp_path: Path,
) -> None:
    """Always-policy evidence remains advisory when RF loses to persistence."""
    spatial_path = _write(
        tmp_path / "spatial.json",
        _spatial_result(interpretable=False, retain_importance=True),
    )

    report = build_evidence_rows(_registry(), spatial_rf_path=spatial_path)
    evidence = report.variable_evidence[0]

    assert evidence.rf_group_importance == 0.2
    assert evidence.rf_importance_interpretable is False
    assert evidence.rf_importance_calculated is True
    assert evidence.recommendation == "retain"


def test_unqualified_vif_and_excluded_eof_are_not_variable_evidence(
    tmp_path: Path,
) -> None:
    """Explicitly false diagnostic flags null values and retain exclusion context."""
    vif_path = _write(
        tmp_path / "vif.json",
        {
            "variable_names": ["era5/2t"],
            "vif_scores": [2.0],
            "evidence_qualified": False,
            "qualification": {"matrix_rank": 34, "predictor_count": 36},
        },
    )
    eof_path = _write(
        tmp_path / "eof.json",
        {
            "variable_names": ["era5/2t"],
            "feature_importance": [0.7],
            "independent_feature_selection_evidence": False,
            "qualification": {"exclusion_reason": "Not independent evidence."},
        },
    )

    report = build_evidence_rows(_registry(), vif_path=vif_path, eof_path=eof_path)
    evidence = report.variable_evidence[0]

    assert evidence.vif is None
    assert evidence.eof_importance is None
    assert report.provenance["diagnostics"]["vif"]["qualified"] is False
    assert (
        report.provenance["diagnostics"]["eof"]["reason"] == "Not independent evidence."
    )


def test_absent_diagnostic_flags_preserve_numeric_evidence(
    tmp_path: Path,
) -> None:
    """Diagnostic JSON from an unqualified (non opt-in) run still contributes evidence."""
    vif_path = _write(
        tmp_path / "vif.json",
        {"variable_names": ["era5/2t"], "vif_scores": [2.0]},
    )
    eof_path = _write(
        tmp_path / "eof.json",
        {"variable_names": ["era5/2t"], "feature_importance": [0.7]},
    )

    report = build_evidence_rows(_registry(), vif_path=vif_path, eof_path=eof_path)
    evidence = report.variable_evidence[0]

    assert evidence.vif == 2.0
    assert evidence.eof_importance == 0.7
    assert report.provenance["diagnostics"]["vif"]["qualified"] is None
    assert "not requested" in report.provenance["diagnostics"]["eof"]["reason"]


def test_parameter_summary_rolls_up_lead_stratum_rows(tmp_path: Path) -> None:
    """The per-parameter rollup aggregates across all lead/stratum evidence rows."""
    spatial_path = _write(
        tmp_path / "spatial.json", _spatial_result(interpretable=True)
    )
    report = build_evidence_rows(_registry(), spatial_rf_path=spatial_path)

    json_path, _, markdown_path = save_evidence_report(report, tmp_path / "report")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert len(payload["parameter_summary"]) == 1
    summary = payload["parameter_summary"][0]
    assert summary["identifier"] == "era5/2t"
    assert summary["lead_stratum_combinations"] == len(report.variable_evidence)
    assert (
        summary["retain_count"]
        + summary["investigate_count"]
        + summary["deprioritise_count"]
        + summary["inconclusive_count"]
        == summary["lead_stratum_combinations"]
    )
    assert summary["overall_recommendation"] == "retain"
    assert "## Parameter summary (rollup across lead and stratum)" in markdown
    assert "era5/2t" in markdown.split("## Parameter summary")[1].split("##")[0]


def test_writes_all_report_formats(tmp_path: Path) -> None:
    """Reports are available for both programmatic and researcher-facing use."""
    report = build_evidence_rows(_registry())

    paths = save_evidence_report(report, tmp_path)

    assert all(path.exists() for path in paths)


def test_report_labels_model_skill_and_provenance_separately(tmp_path: Path) -> None:
    """Machine and Markdown outputs distinguish qualification from contribution."""
    spatial_path = _write(
        tmp_path / "spatial.json", _spatial_result(interpretable=True)
    )
    report = build_evidence_rows(_registry(), spatial_rf_path=spatial_path)

    json_path, _, markdown_path = save_evidence_report(report, tmp_path / "report")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["provenance"] == {
        "effective_samples": 40,
        "feature_count": 2,
        "initialisation_count": 10,
        "target_mode": "absolute",
        "rf_settings": {"n_estimators": 20, "permutation_repeats": 2},
    }
    assert payload["variable_evidence"][0]["rf_mse_improvement"] is None
    assert payload["model_qualification"][1]["mse_improvement"] == 0.1
    assert "not confirmatory evidence" in payload["warning"]
    assert "## Model qualification (RF versus persistence)" in markdown
    assert "## Variable-level evidence" in markdown
    assert "Mean RF MSE improvement" not in markdown


def test_markdown_reports_diagnostic_qualification_reason(tmp_path: Path) -> None:
    """Diagnostic qualification status and reason are researcher-visible."""
    eof_path = _write(
        tmp_path / "eof.json",
        {
            "variable_names": ["era5/2t"],
            "feature_importance": [0.7],
            "independent_feature_selection_evidence": False,
            "qualification": {"exclusion_reason": "Not independent evidence."},
        },
    )
    report = build_evidence_rows(_registry(), eof_path=eof_path)

    _, _, markdown_path = save_evidence_report(report, tmp_path / "report")
    markdown = markdown_path.read_text(encoding="utf-8")

    assert (
        "EOF evidence qualified: false; reason: Not independent evidence." in markdown
    )
