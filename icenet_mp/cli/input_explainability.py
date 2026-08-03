r"""``imp input-explainability`` — Run supervised input variable explainability together.

Executes Random Forest feature importance analysis on the configured dataset variables,
writing all results into a single output directory with subdirectories for each strand.

This is a **supervised** analysis: it trains a model to predict the target variable (SIC)
from all other inputs, then derives per-feature importance from permutation scores. It helps
identify which input variables are most predictive of sea-ice concentration.

Each RF sample is built from a temporal window of ``n_history_steps`` days of input history
followed by ``n_forecast_steps`` forecast days for the target — matching how the model is
trained via ``CombinedDataset``. This replaces the old single-day sampling approach that
ignored temporal structure.

Future additions may include SHAP values and partial dependence plots.

Usage::

    imp input-explainability --config-name explainability/rf \\
        'data/datasets=[full_sicnorth_ssmis_25p0km-1979-2024-24h-v2]' \\
        ++vif.max_samples=1000

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from omegaconf import DictConfig  # noqa: TC002

from icenet_mp.cli.plotting import maybe_plot_spatial_rf_results
from icenet_mp.input_diagnostics.data import (
    _get_max_samples,
    build_datasets,
    resolve_datasets,
)

from .hydra import hydra_adaptor

if TYPE_CHECKING:
    from icenet_mp.data_loaders.single_dataset import SingleDataset

logger = logging.getLogger(__name__)

input_exp_app = typer.Typer(
    help=(
        "Run supervised input explainability (Random Forest) and output results to a single directory."
    ),
)


def _run_rf_strand(
    datasets: dict[str, SingleDataset],
    config: DictConfig,
    output_dir: Path,
) -> None:
    """Run Random Forest feature importance and save results using temporal windows."""
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

    # Build temporal windows.
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

    logger.info("Running Random Forest feature importance...")
    interaction_enabled = (rf_config.get("interaction") or {}).get("enabled", True)
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


def _run_all_strands(
    config: DictConfig,
    output_dir: Path,
) -> None:
    """Run Random Forest explainability into subdirectories.

    Args:
        config: Hydra-composed config.
        output_dir: Root directory for all outputs (subdirs created automatically).

    """
    logger.info("Building dataset instances for input explainability...")
    group_paths, group_variables = resolve_datasets(config)

    datasets = build_datasets(group_paths, group_variables)

    # Run RF strand (uses temporal windows internally).
    try:
        _run_rf_strand(datasets, config, output_dir)
    except Exception:
        logger.exception("RF explainability failed")

    typer.echo(f"\nAll explainability complete. Results in {output_dir}/")


@input_exp_app.callback(invoke_without_command=True)
@hydra_adaptor
def input_explainability(
    _ctx: typer.Context,
    config: DictConfig,
    output_dir: Annotated[
        str,
        typer.Option(
            help="Root directory for all explainability results (subdirs: rf/).",
        ),
    ] = "outputs/input_explainability",
) -> None:
    """Run supervised input explainability (Random Forest)."""
    _run_all_strands(config, Path(output_dir))
