r"""``imp input-diagnostics`` — Run unsupervised input variable diagnostics together.

Executes VIF, PCA, and EOF analyses on the configured dataset variables, writing all
results into a single output directory with subdirectories for each strand.

These are **unsupervised** analyses: they examine variance structure and redundancy in
the input features alone, without reference to the target (SIC). They help identify:

- **VIF**: Variables that can be predicted from others (multicollinearity).
- **PCA**: Directions of maximum variance; weighted loadings show which variables drive each direction.
- **EOF**: Geoscience-style decomposition into orthogonal spatial patterns ranked by explained variance.

Usage::

    imp input-diagnostics --config-name vif \\
        'data/datasets=[full_sicnorth_ssmis_25p0km-1979-2024-24h-v2]' \\
        ++vif.max_samples=1000

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from omegaconf import DictConfig  # noqa: TC002

from icenet_mp.data_loaders.single_dataset import SingleDataset
from icenet_mp.input_diagnostics.data import (
    _get_max_samples,
    build_sample_matrix,
    resolve_datasets,
)

from .hydra import hydra_adaptor

if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)

input_diag_app = typer.Typer(
    help=(
        "Run unsupervised input diagnostics (VIF, PCA, EOF) together and output "
        "results to a single directory."
    ),
)


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
    try:
        result = compute_vif(sample_matrix, var_names, threshold=threshold)
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


def _run_all_strands(
    config: DictConfig,
    output_dir: Path,
) -> None:
    """Run VIF, PCA, and EOF analyses into subdirectories.

    Args:
        config: Hydra-composed config.
        output_dir: Root directory for all outputs (subdirs created automatically).

    """
    max_samples = _get_max_samples(config, "vif")
    # Each module can override via its own namespace (e.g. pca.max_samples).
    # Use the most restrictive value across all modules to ensure no strand
    # receives more data than any configured limit.
    for mod in ("pca", "eof"):
        ms = _get_max_samples(config, mod)
        if ms is not None:
            max_samples = min(max_samples, ms) if max_samples is not None else ms

    threshold = float(config.get("vif", {}).get("threshold", 5.0))

    logger.info("Building dataset instances for input diagnostics...")
    group_paths, group_variables = resolve_datasets(config)

    datasets: dict[str, SingleDataset] = {}
    for group_name, paths in group_paths.items():
        vars_list = group_variables.get(group_name)
        if vars_list is not None and len(vars_list) > 0:
            logger.info(
                "Loading dataset group %r (%d paths, variables: %s).",
                group_name,
                len(paths),
                vars_list,
            )
            datasets[group_name] = SingleDataset(
                group_name,
                paths,
                variables=vars_list,
            )
        else:
            logger.warning(
                "Dataset group %r has no variable filter — loading all variables (%d paths). "
                "Consider specifying 'variables' to limit data loaded.",
                group_name,
                len(paths),
            )
            datasets[group_name] = SingleDataset(group_name, paths)

    sample_matrix, var_names = build_sample_matrix(datasets, max_samples=max_samples)
    logger.info(
        "Sample matrix shape: %s (%d variables).", sample_matrix.shape, len(var_names)
    )

    # Run each strand; failures in one do not prevent others from running.
    for name, fn in [
        (
            "VIF",
            lambda: _run_vif_strand(sample_matrix, var_names, threshold, output_dir),
        ),
        ("PCA", lambda: _run_pca_strand(sample_matrix, var_names, output_dir)),
        ("EOF", lambda: _run_eof_strand(sample_matrix, var_names, output_dir)),
    ]:
        try:
            fn()
        except Exception:
            logger.exception("%s analysis failed", name)

    typer.echo(f"\nAll diagnostics complete. Results in {output_dir}/")


@input_diag_app.callback(invoke_without_command=True)
@hydra_adaptor
def input_diagnostics(
    _ctx: typer.Context,
    config: DictConfig,
    output_dir: Annotated[
        str,
        typer.Option(
            help="Root directory for all diagnostic results (subdirs: vif/, pca/, eof/).",
        ),
    ] = "outputs/input_diagnostics",
) -> None:
    """Run VIF, PCA, and EOF unsupervised diagnostics together."""
    _run_all_strands(config, Path(output_dir))
