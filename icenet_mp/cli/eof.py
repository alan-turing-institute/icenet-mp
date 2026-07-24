r"""``imp eof`` — Empirical Orthogonal Function analysis for input variable feature importance.

Computes EOF on spatially-aggregated climate data — mathematically equivalent to PCA but
with geoscience conventions: modes (patterns), coefficients (time series), and variance
explained per mode. Derives per-variable importance from variance-weighted absolute loadings.
Results are written to an output directory as JSON + text report.

Usage::

    imp eof --config-name baseline/03_cnn_vit_cnn \\
        'data/datasets=[full_sicnorth_ssmis_25p0km-1979-2024-24h-v2]'

"""

from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.input_diagnostics.eof import (
    print_eof_table,
    run_eof_analysis,
    save_eof_results,
)

from .hydra import hydra_adaptor

eof_app = typer.Typer(
    help="Empirical Orthogonal Function analysis for input variable feature importance.",
)


@eof_app.command()
@hydra_adaptor
def eof(
    config: DictConfig,
    output_dir: Annotated[
        str,
        typer.Option(
            help="Directory to write EOF results (JSON + text report) to.",
        ),
    ] = "outputs/eof",
) -> None:
    """Run EOF analysis on the configured dataset variables."""
    result = run_eof_analysis(config)
    print_eof_table(result)

    json_path = save_eof_results(result, Path(output_dir))
    typer.echo(f"Results written to {json_path}")
