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

import numpy as np
import typer
from omegaconf import (
    DictConfig,  # noqa: TC002 — needed at runtime for @hydra_adaptor annotation resolution
)

if TYPE_CHECKING:
    from icenet_mp.data_loaders.single_dataset import SingleDataset

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
) -> None:
    """Run VIF analysis and save results."""
    from icenet_mp.input_diagnostics.vif import (  # noqa: PLC0415
        compute_vif,
        print_vif_table,
        save_vif_results,
    )

    logger.info("Running VIF analysis...")
    result = compute_vif(sample_matrix, var_names, threshold=threshold)
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
    result = compute_pca(sample_matrix, var_names)
    print_pca_table(result)
    json_path = save_pca_results(result, output_dir / "pca")
    typer.echo(f"PCA results written to {json_path}")


def _run_eof_strand(
    sample_matrix: np.ndarray,
    var_names: list[str],
    output_dir: Path,
) -> None:
    """Run EOF analysis and save results."""
    from icenet_mp.input_diagnostics.eof import (  # noqa: PLC0415
        compute_eof,
        print_eof_table,
        save_eof_results,
    )

    logger.info("Running EOF analysis...")
    result = compute_eof(sample_matrix, var_names)
    print_eof_table(result)
    json_path = save_eof_results(result, output_dir / "eof")
    typer.echo(f"EOF results written to {json_path}")


def _run_rf_strand(  # noqa: C901, PLR0912, PLR0915 — complex data-loading + RF pipeline
    datasets: dict[str, SingleDataset],
    group_paths: dict[str, list[Path]],
    config: DictConfig,
    output_dir: Path,
) -> None:
    """Run Random Forest feature importance using temporal windows and save results."""
    from icenet_mp.input_explainability.rf import (  # noqa: PLC0415
        _get_rf_window_params,
        _windows_to_arrays,
        build_rf_windows,
        compute_rf_importance,
        print_rf_table,
        save_rf_results,
    )

    rf_config = config.get("rf", {}) or {}
    n_jobs = int(rf_config.get("n_jobs", -1))

    # Resolve temporal window parameters.
    n_history_steps, n_forecast_steps = _get_rf_window_params(config)
    logger.info(
        "RF temporal windows: history=%d steps, forecast=%d steps.",
        n_history_steps, n_forecast_steps,
    )

    max_samples = _get_max_samples(config, "rf")

    # Load target variable (SIC/ice_conc).
    target_cfg = rf_config.get("target", {}) or {}
    target_group_as = str(target_cfg.get("group_as", "sic-ssmis"))
    target_variable = str(target_cfg.get("variable", "ice_conc"))

    logger.info(
        "Loading target variable %r from group %r.",
        target_variable, target_group_as,
    )

    # Find the zarr file for this group — look up by resolved group name first.
    target_path: Path | None = None
    if target_group_as in group_paths:
        paths_for_group = group_paths[target_group_as]
        if paths_for_group:
            target_path = paths_for_group[0]

    # Fallback: scan for any zarr whose path contains the target group name.
    if target_path is None:
        for paths in group_paths.values():
            for p in paths:
                if target_group_as in str(p):
                    target_path = p
                    break
            if target_path is not None:
                break

    if target_path is None:
        logger.warning("Target dataset %r not found; skipping RF strand.", target_group_as)
        return

    # Build temporal windows — no explicit target path means first dataset is used.
    try:
        windows, var_names = build_rf_windows(
            datasets,
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
    X, y = _windows_to_arrays(windows, var_names, n_history_steps)  # noqa: N806 — standard ML convention for feature matrix

    logger.info(
        "Final arrays — features: %s, target (%s): %s", X.shape, target_variable, y.shape,
    )

    # Prevent data leakage: if the target variable is also present as a feature column,
    # remove all columns corresponding to it across all history steps.
    target_label = f"{target_group_as}/{target_variable}"
    leak_indices: list[int] = []
    new_var_names: list[str] = []

    n_vars_per_step = len(var_names)  # variables per history step
    total_feature_cols = n_vars_per_step * n_history_steps

    for col in range(total_feature_cols):
        var_idx_in_step = col % n_vars_per_step
        candidate_name = var_names[var_idx_in_step] if var_idx_in_step < len(var_names) else ""
        if candidate_name == target_label:
            leak_indices.append(col)
        else:
            new_var_names.append(candidate_name)

    if leak_indices:
        logger.info(
            "Removing %d data-leakage column(s) for feature %r across all history steps.",
            len(leak_indices), target_label,
        )
        X = np.delete(X, leak_indices, axis=1)  # noqa: N806 — standard ML convention for feature matrix

    logger.info("Running Random Forest feature importance...")
    result = compute_rf_importance(
        X, y, new_var_names, target_name=f"{target_group_as}/{target_variable}",
        n_estimators=rf_config.get("n_estimators", 500),
        max_depth=rf_config.get("max_depth", 15),
        min_samples_leaf=rf_config.get("min_samples_leaf", 5),
        min_samples_split=rf_config.get("min_samples_split", 8),
        max_features=rf_config.get("max_features", "sqrt"),
        random_state=rf_config.get("random_state", 42),
        n_jobs=n_jobs,
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
    max_samples = _get_max_samples(config, "vif")
    # Each module can override via its own namespace (e.g. pca.max_samples).
    # Use the most restrictive value across all modules to ensure no strand
    # receives more data than any configured limit.
    for mod in ("pca", "eof", "rf"):
        ms = _get_max_samples(config, mod)
        if ms is not None:
            max_samples = min(max_samples, ms) if max_samples is not None else ms

    threshold = float(config.get("vif", {}).get("threshold", 5.0))

    logger.info("Building dataset instances for pre-feature analysis...")
    group_paths, _ = resolve_datasets(config)
    datasets = build_datasets(group_paths, {})

    sample_matrix, var_names = build_sample_matrix(datasets, max_samples=max_samples)
    logger.info("Sample matrix shape: %s (%d variables).", sample_matrix.shape, len(var_names))

    # Run each strand; failures in one do not prevent others from running.
    for name, fn in [
        ("VIF", lambda: _run_vif_strand(sample_matrix, var_names, threshold, output_dir)),
        ("PCA", lambda: _run_pca_strand(sample_matrix, var_names, output_dir)),
        ("EOF", lambda: _run_eof_strand(sample_matrix, var_names, output_dir)),
        (
            "RF",
            lambda: _run_rf_strand(datasets, group_paths, config, output_dir),
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
