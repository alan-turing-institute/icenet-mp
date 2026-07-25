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
from typing import Annotated

import numpy as np
import typer
from omegaconf import DictConfig  # noqa: TC002

from icenet_mp.data_loaders.single_dataset import SingleDataset
from icenet_mp.input_diagnostics.data import (
    _get_max_samples,
    resolve_datasets,
)

from .hydra import hydra_adaptor

logger = logging.getLogger(__name__)

input_exp_app = typer.Typer(
    help=(
        "Run supervised input explainability (Random Forest) and output results to a single directory."
    ),
)


def _run_rf_strand(  # noqa: C901, PLR0912, PLR0915 — complex data-loading + RF pipeline
    datasets: dict[str, SingleDataset],
    group_paths: dict[str, list[Path]],
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

    # Build temporal windows.
    try:
        windows, var_names = build_rf_windows(
            datasets, target_path=target_path, target_variable=target_variable,
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

    # Run RF strand (uses temporal windows internally).
    try:
        _run_rf_strand(datasets, group_paths, config, output_dir)
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
