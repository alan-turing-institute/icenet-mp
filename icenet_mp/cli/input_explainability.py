r"""``imp input-explainability`` — Run supervised input variable explainability together.

Executes Random Forest feature importance analysis on the configured dataset variables,
writing all results into a single output directory with subdirectories for each strand.

This is a **supervised** analysis: it trains a model to predict the target variable (SIC)
from all other inputs, then derives per-feature importance from permutation scores. It helps
identify which input variables are most predictive of sea-ice concentration.

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
    build_sample_matrix,
    resolve_datasets,
)

from .hydra import hydra_adaptor

logger = logging.getLogger(__name__)

input_exp_app = typer.Typer(
    help=(
        "Run supervised input explainability (Random Forest) and output results to a single directory."
    ),
)


def _run_rf_strand(
    sample_matrix: np.ndarray,
    var_names: list[str],
    datasets: dict[str, SingleDataset],
    config: DictConfig,
    output_dir: Path,
) -> None:
    """Run Random Forest feature importance and save results."""
    from icenet_mp.input_explainability.rf import (  # noqa: PLC0415
        compute_rf_importance,
        print_rf_table,
        save_rf_results,
    )

    rf_config = config.get("rf", {})
    n_jobs = int(rf_config.get("n_jobs", 1))

    # Work on copies so mutations (data-leakage removal) don't affect callers.
    X = sample_matrix.copy()  # noqa: N806 — standard ML convention for feature matrix
    names = list(var_names)

    # Extract target variable (SIC/ice_conc) from datasets.
    target_cfg = rf_config.get("target", {})
    target_group_as = str(target_cfg.get("group_as", "sic-ssmis"))
    target_variable = str(target_cfg.get("variable", "ice_conc"))

    # Find the dataset that contains the target variable.
    target_ds: SingleDataset | None = None
    for group_name, ds in datasets.items():
        if target_group_as in group_name or target_group_as in ds.name:
            target_ds = ds
            break

    if target_ds is None:
        logger.warning("Target dataset %r not found; skipping RF strand.", target_group_as)
        return

    # Get aligned dates (same as used for sample_matrix).
    common_dates = sorted(set.intersection(*(set(ds.dates) for ds in datasets.values())))
    max_samples = _get_max_samples(config, "rf")
    if max_samples is not None and max_samples < len(common_dates):
        indices = np.linspace(0, len(common_dates) - 1, max_samples, dtype=int)
        common_dates = [common_dates[i] for i in indices]

    # Build target array from aligned dates — extract only the target variable's channel.
    y: list[float] = []
    try:
        target_channel_idx = target_ds.variable_names.index(target_variable)
    except ValueError:
        logger.warning(
            "Target variable %r not found in dataset %s (variables: %s). "
            "Falling back to mean across all channels.",
            target_variable, target_ds.name, target_ds.variable_names,
        )
        target_channel_idx = None

    for date in common_dates:
        tchw = target_ds.get_tchw([date])  # shape [T=1, C, H, W]
        if target_channel_idx is not None and target_channel_idx < tchw.shape[1]:
            spatial_mean = float(tchw[:, target_channel_idx].mean())
        else:
            spatial_mean = float(tchw.mean())
        y.append(spatial_mean)

    y_arr = np.array(y)

    # Prevent data leakage: if the target variable is also present as a feature column,
    # remove it from X.  This can happen when the SIC dataset appears in both the input
    # datasets list and as the target group (the common case for explainability).
    target_label = f"{target_group_as}/{target_variable}"
    if target_label in names:
        leak_idx = names.index(target_label)
        logger.info(
            "Removing data-leakage feature %r from X (it is also the prediction target).",
            target_label,
        )
        X = np.delete(X, leak_idx, axis=1)  # noqa: N806 — standard ML convention for feature matrix
        names.pop(leak_idx)

    logger.info("Running Random Forest feature importance...")
    result = compute_rf_importance(
        X, y_arr, names, target_name=f"{target_group_as}/{target_variable}",
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
    max_samples = _get_max_samples(config, "rf")

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

    sample_matrix, var_names = build_sample_matrix(datasets, max_samples=max_samples)
    logger.info("Sample matrix shape: %s (%d variables).", sample_matrix.shape, len(var_names))

    # Run RF strand.
    try:
        _run_rf_strand(sample_matrix, var_names, datasets, config, output_dir)
    except Exception:
        logger.exception("RF explainability failed")

    typer.echo(f"\nAll explainability complete. Results in {output_dir}/")


@input_exp_app.command(name="run")
@hydra_adaptor
def input_explainability(
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
