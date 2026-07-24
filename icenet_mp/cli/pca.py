r"""``imp pca`` — Principal Component Analysis for input variable feature importance.

Standardises variables, computes principal components via SVD, and derives per-variable
importance scores from weighted absolute loadings across all components. Results are
written to an output directory as JSON + text report.

Usage::

    imp pca --config-name baseline/03_cnn_vit_cnn \\
        'data/datasets=[full_sicnorth_ssmis_25p0km-1979-2024-24h-v2]'

"""

from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.input_diagnostics.pca import (
    print_pca_table,
    run_pca_analysis,
    save_pca_results,
)

from .hydra import hydra_adaptor

pca_app = typer.Typer(
    help="Principal Component Analysis for input variable feature importance.",
)


@pca_app.command()
@hydra_adaptor
def pca(
    config: DictConfig,
    output_dir: Annotated[
        str,
        typer.Option(
            help="Directory to write PCA results (JSON + text report) to.",
        ),
    ] = "outputs/pca",
) -> None:
    """Run PCA analysis on the configured dataset variables."""
    result = run_pca_analysis(config)
    print_pca_table(result)

    json_path = save_pca_results(result, Path(output_dir))
    typer.echo(f"Results written to {json_path}")
