r"""``imp pre-feature-analysis`` — Run all input variable analysis strands together.

Executes VIF, PCA, EOF, Random Forest feature importance, and correlation heatmap
analyses on the configured dataset variables, writing all results into a single output
directory with subdirectories for each strand.

Usage::

    imp pre-feature-analysis --config-name baseline/03_cnn_vit_cnn \
        'data/datasets=[full_sicnorth_ssmis_25p0km-1979-2024-24h-v2]' \
        ++vif.max_samples=1000

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from omegaconf import (
    DictConfig,  # noqa: TC002 — needed at runtime for @hydra_adaptor annotation resolution
)

if TYPE_CHECKING:
    import numpy as np

    from icenet_mp.data_loaders.single_dataset import SingleDataset

from icenet_mp.cli.plotting import maybe_plot_spatial_rf_results
from icenet_mp.input_diagnostics.data import (
    _get_max_samples,
    build_datasets,
    build_sample_matrix,
    resolve_datasets,
)

from .hydra import hydra_adaptor

logger = logging.getLogger(__name__)

pre_feature_app = typer.Typer(
    help=(
        "Run all input variable analysis strands (VIF, PCA, EOF, RF, Heatmap) "
        "together and output results to a single directory."
    ),
)


def _build_datasets_from_config(config: DictConfig) -> dict[str, SingleDataset]:
    """Build SingleDataset instances from Hydra config (shared across all strands).

    Delegates to :func:`input_diagnostics.data.resolve_datasets` and
    :func:`input_diagnostics.data.build_datasets` so that VIF, PCA, EOF, RF,
    and Heatmap all draw from the same data without re-parsing.

    Args:
        config: Hydra-composed config.

    Returns:
        Mapping of group name to SingleDataset instance.

    """
    group_paths, group_variables = resolve_datasets(config)
    return build_datasets(group_paths, group_variables)


def _run_vif_strand(
    sample_matrix: np.ndarray,
    var_names: list[str],
    threshold: float,
    output_dir: Path,
    qualification: dict[str, object] | None = None,
) -> None:
    """Run VIF analysis and save results."""
    from icenet_mp.input_diagnostics.vif import (  # noqa: PLC0415
        compute_vif,
        print_vif_table,
        save_vif_results,
    )

    logger.info("Running VIF analysis...")
    try:
        q = qualification or {}
        result = compute_vif(
            sample_matrix,
            var_names,
            threshold=threshold,
            qualification_enabled=bool(q.get("enabled", False)),
            minimum_sample_feature_ratio=float(
                str(q.get("minimum_sample_feature_ratio", 10.0))
            ),
            date_range=q.get("date_range"),  # type: ignore[arg-type]
            sampling_limit=q.get("sampling_limit"),  # type: ignore[arg-type]
        )
    except ValueError as exc:
        typer.echo(f"VIF skipped: {exc}", err=True)
        return
    print_vif_table(result)
    json_path = save_vif_results(result, output_dir / "vif")
    typer.echo(f"VIF results written to {json_path}")


def _run_pca_strand(
    sample_matrix: np.ndarray,
    var_names: list[str],
    output_dir: Path,
) -> None:
    """Run PCA analysis and save results."""
    from icenet_mp.input_diagnostics.pca import (  # noqa: PLC0415
        compute_pca,
        print_pca_table,
        save_pca_results,
    )

    logger.info("Running PCA analysis...")
    try:
        result = compute_pca(sample_matrix, var_names)
    except ValueError as exc:
        typer.echo(f"PCA skipped: {exc}", err=True)
        return
    print_pca_table(result)
    json_path = save_pca_results(result, output_dir / "pca")
    typer.echo(f"PCA results written to {json_path}")


def _run_eof_strand(
    sample_matrix: np.ndarray,
    var_names: list[str],
    output_dir: Path,
    qualification: dict[str, object] | None = None,
) -> None:
    """Run EOF analysis and save results."""
    from icenet_mp.input_diagnostics.eof import (  # noqa: PLC0415
        compute_eof,
        print_eof_table,
        save_eof_results,
    )

    logger.info("Running EOF analysis...")
    q = qualification or {}
    result = compute_eof(
        sample_matrix,
        var_names,
        qualification_enabled=bool(q.get("enabled", False)),
        date_range=q.get("date_range"),  # type: ignore[arg-type]
        sampling_limit=q.get("sampling_limit"),  # type: ignore[arg-type]
    )
    print_eof_table(result)
    json_path = save_eof_results(result, output_dir / "eof")
    typer.echo(f"EOF results written to {json_path}")


def _run_rf_strand(
    datasets: dict[str, SingleDataset],
    config: DictConfig,
    output_dir: Path,
) -> None:
    from icenet_mp.input_explainability.rf import (  # noqa: PLC0415
        _get_rf_window_params,
        _windows_to_arrays,
        build_rf_windows,
        compute_rf_importance,
        print_rf_table,
        save_rf_results,
    )

    rf_config = config.get("rf", {}) or {}
    if rf_config.get("mode", "scalar") == "spatial":
        from icenet_mp.input_explainability.rf import run_rf_analysis  # noqa: PLC0415
        from icenet_mp.input_explainability.spatial_rf import (  # noqa: PLC0415
            SpatialRFResult,
            save_spatial_rf_results,
        )

        result = run_rf_analysis(config)
        if not isinstance(result, SpatialRFResult):
            msg = "Spatial RF configuration did not produce a sampled-map result."
            raise RuntimeError(msg)
        spatial_json_path, _ = save_spatial_rf_results(result, output_dir / "rf")
        typer.echo(f"Sampled-map screening results written to {spatial_json_path}")
        maybe_plot_spatial_rf_results(config, result, output_dir / "rf")
        return
    n_jobs = int(rf_config.get("n_jobs", -1))

    # Resolve temporal window parameters.
    n_history_steps, n_forecast_steps = _get_rf_window_params(config)
    logger.info(
        "RF temporal windows: history=%d steps, forecast=%d steps.",
        n_history_steps,
        n_forecast_steps,
    )

    max_samples = _get_max_samples(config, "rf")

    # Load target variable (SIC/ice_conc).
    target_cfg = rf_config.get("target", {}) or {}
    target_group_as = str(target_cfg.get("group_as", "sic-ssmis"))
    target_variable = str(target_cfg.get("variable", "ice_conc"))

    logger.info(
        "Loading target variable %r from group %r.",
        target_variable,
        target_group_as,
    )

    # The configured target group must be explicitly present; approximate name matching
    # could silently select the wrong physical dataset.
    target_ds = datasets.get(target_group_as)
    if target_ds is None:
        msg = (
            f"Configured RF target group {target_group_as!r} is not included in the "
            "resolved datasets."
        )
        raise ValueError(msg)

    train_ranges = config.get("data", {}).get("split", {}).get("train")
    if not train_ranges:
        msg = "RF analysis requires configured data.split.train date ranges."
        raise ValueError(msg)
    datasets = {
        name: ds.subset(date_ranges=list(train_ranges)) for name, ds in datasets.items()
    }
    target_ds = datasets[target_group_as]

    # Build temporal windows — target_ds ensures the correct variable is used for the target.
    try:
        windows, var_names = build_rf_windows(
            datasets,
            target_ds=target_ds,
            target_variable=target_variable,
            n_history_steps=n_history_steps,
            n_forecast_steps=n_forecast_steps,
            max_samples=max_samples,
        )
    except ValueError as exc:
        logger.warning("RF window building failed: %s", exc)
        return

    if not windows:
        logger.warning("No valid RF windows found; skipping RF strand.")
        return

    # Convert windows to feature matrix and target vector.
    X, y, expanded_names = _windows_to_arrays(windows, var_names, n_history_steps)  # noqa: N806 — standard ML convention for feature matrix

    logger.info(
        "Final arrays — features: %s, target (%s): %s",
        X.shape,
        target_variable,
        y.shape,
    )

    # Historical target observations are valid inputs, not leakage.

    interaction_enabled = (rf_config.get("interaction") or {}).get("enabled", True)
    logger.info("Running Random Forest feature importance...")
    result = compute_rf_importance(
        X,
        y,
        expanded_names,
        target_name=f"{target_group_as}/{target_variable}",
        n_estimators=rf_config.get("n_estimators", 500),
        max_depth=rf_config.get("max_depth", 15),
        min_samples_leaf=rf_config.get("min_samples_leaf", 5),
        min_samples_split=rf_config.get("min_samples_split", 8),
        max_features=rf_config.get("max_features", "sqrt"),
        random_state=rf_config.get("random_state", 42),
        n_jobs=n_jobs,
        interaction_enabled=interaction_enabled,
    )
    print_rf_table(result)
    json_path = save_rf_results(result, output_dir / "rf")
    typer.echo(f"RF results written to {json_path}")


def _run_correlation_strand(
    sample_matrix: np.ndarray,
    var_names: list[str],
    output_dir: Path,
) -> None:
    """Run correlation heatmap analysis and save results."""
    from icenet_mp.input_diagnostics.correlation import (  # noqa: PLC0415
        compute_correlation_matrix,
        plot_correlation_heatmap,
        print_correlation_summary,
        save_correlation_csv,
    )

    logger.info("Running correlation heatmap...")
    corr_df = compute_correlation_matrix(sample_matrix, var_names)
    print_correlation_summary(corr_df)
    plot_correlation_heatmap(corr_df, output_dir / "correlations")
    save_correlation_csv(corr_df, output_dir / "correlations")
    typer.echo(f"Correlation heatmap written to {output_dir / 'correlations'}")


def _run_all_strands(
    config: DictConfig,
    output_dir: Path,
) -> None:
    """Run VIF, PCA, EOF, RF, and Heatmap analyses into subdirectories.

    Args:
        config: Hydra-composed config.
        output_dir: Root directory for all outputs (subdirs created automatically).

    """
    threshold = float(config.get("vif", {}).get("threshold", 5.0))

    logger.info("Building dataset instances for pre-feature analysis...")
    group_paths, group_variables = resolve_datasets(config)
    datasets = build_datasets(group_paths, group_variables)

    registry = None
    if (config.get("feature_evidence", {}) or {}).get("registry", {}).get("entries"):
        from icenet_mp.feature_evidence.registry import (  # noqa: PLC0415
            load_feature_registry,
        )

        registry = load_feature_registry(config)
        registry.validate_available(
            {group: dataset.variable_names for group, dataset in datasets.items()}
        )

    matrices = {
        module: build_sample_matrix(
            datasets, max_samples=_get_max_samples(config, module)
        )
        for module in ("vif", "pca", "eof")
    }
    sample_matrix, var_names = matrices["vif"]
    logger.info(
        "Sample matrix shape: %s (%d variables).", sample_matrix.shape, len(var_names)
    )

    # Run each strand; retain failures so a sampled-map screening failure cannot be
    # mistaken for a successful completed analysis.
    failures: list[str] = []
    vif_cfg = config.get("vif", {}) or {}
    qualification_cfg = dict(vif_cfg.get("qualification", {}) or {})
    if qualification_cfg.get("enabled", False):
        common_dates = sorted(
            set.intersection(*(set(ds.dates) for ds in datasets.values()))
        )
        qualification_cfg.update(
            {
                "date_range": (str(common_dates[0]), str(common_dates[-1])),
                "sampling_limit": _get_max_samples(config, "vif"),
            }
        )
    pca_matrix, pca_names = matrices["pca"]
    eof_matrix, eof_names = matrices["eof"]
    for name, fn in [
        (
            "VIF",
            lambda: _run_vif_strand(
                sample_matrix, var_names, threshold, output_dir, qualification_cfg
            ),
        ),
        ("PCA", lambda: _run_pca_strand(pca_matrix, pca_names, output_dir)),
        (
            "EOF",
            lambda: _run_eof_strand(
                eof_matrix, eof_names, output_dir, qualification_cfg
            ),
        ),
        (
            "RF",
            lambda: _run_rf_strand(datasets, config, output_dir),
        ),
        (
            "Correlation heatmap",
            lambda: _run_correlation_strand(sample_matrix, var_names, output_dir),
        ),
    ]:
        try:
            fn()
        except Exception:
            logger.exception("%s analysis failed", name)
            failures.append(name)

    if failures:
        typer.echo(
            f"\nAnalyses completed with failures ({', '.join(failures)}). Results in {output_dir}/"
        )
        if (config.get("rf", {}) or {}).get(
            "mode", "scalar"
        ) == "spatial" and "RF" in failures:
            msg = "Sampled-map RF screening failed; no feature-selection result is available."
            raise RuntimeError(msg)
        return
    if registry is not None:
        from icenet_mp.feature_evidence.report import (  # noqa: PLC0415
            build_evidence_rows,
            save_evidence_report,
        )

        evidence_paths = save_evidence_report(
            build_evidence_rows(
                registry,
                vif_path=output_dir / "vif" / "vif_results.json",
                pca_path=output_dir / "pca" / "pca_results.json",
                eof_path=output_dir / "eof" / "eof_results.json",
                correlation_path=output_dir / "correlations" / "correlations.csv",
                spatial_rf_path=output_dir / "rf" / "spatial_rf_results.json",
            ),
            output_dir / "evidence",
        )
        typer.echo(f"Feature evidence report written to {evidence_paths[0]}")
    typer.echo(f"\nAll analyses complete. Results in {output_dir}/")


@pre_feature_app.callback(invoke_without_command=True)
@hydra_adaptor
def pre_feature_analysis(
    _ctx: typer.Context,
    config: DictConfig,
    output_dir: Annotated[
        str,
        typer.Option(
            help="Root directory for all analysis results (subdirs: vif/, pca/, eof/, rf/, correlations/).",
        ),
    ] = "outputs/pre_feature_analysis",
) -> None:
    """Run VIF, PCA, EOF, RF feature importance, and correlation heatmap together."""
    _run_all_strands(config, Path(output_dir))
