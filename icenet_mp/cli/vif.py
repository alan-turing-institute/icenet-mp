r"""``imp vif`` — Variance Inflation Factor analysis for input variable multicollinearity.

Loads dataset variables from zarr, computes per-variable VIF scores, and reports
multicollinearity (threshold configurable via Hydra config). Results are written to
an output directory as JSON + text report.

Usage::

    imp vif --config-name baseline/03_cnn_vit_cnn \\
        'data/datasets=[full_sicnorth_ssmis_25p0km_1979_2024_24h_v2]' \\
        '+data.datasets.full-sicnorth-ssmis-25p0km-1979-2024-24h-v2.variables=[ice_conc]'

"""

from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.input_diagnostics.vif import (
    print_vif_table,
    run_vif_analysis,
    save_vif_results,
)

from .hydra import hydra_adaptor

vif_app = typer.Typer(
    help="Variance Inflation Factor analysis for input variable multicollinearity.",
)


@vif_app.command()
@hydra_adaptor
def vif(
    config: DictConfig,
    output_dir: Annotated[
        str,
        typer.Option(
            help="Directory to write VIF results (JSON + text report) to.",
        ),
    ] = "outputs/vif",
) -> None:
    """Run VIF analysis on the configured dataset variables."""
    result = run_vif_analysis(config)
    print_vif_table(result)

    json_path = save_vif_results(result, Path(output_dir))
    typer.echo(f"Results written to {json_path}")
