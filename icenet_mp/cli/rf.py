r"""``imp rf`` — Random Forest feature importance for input explainability.

Trains a single Random Forest regressor to predict the target variable (e.g., next-day SIC)
from all other input variables, then derives per-feature importance from permutation-based
scores on a held-out test set. Also computes pairwise interaction strengths between features.

**Interpretation:** Permutation importance measures how much each feature contributes to
predicting the target — high importance means the model relies heavily on that variable.
Interaction scores reveal synergistic or redundant relationships between pairs of features.

Results are written to an output directory as JSON + text report, plus visualisation plots.

Usage::

    imp rf --config-name explainability/rf \\
        'data/datasets=[full_sicnorth_ssmis_25p0km-1979-2024-24h-v2]'

"""

from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.input_explainability.rf import (
    print_rf_table,
    run_rf_analysis,
    save_rf_results,
)

from .hydra import hydra_adaptor

rf_app = typer.Typer(
    help="Random Forest feature importance for input variable analysis.",
)


@rf_app.command()
@hydra_adaptor
def rf(
    config: DictConfig,
    output_dir: Annotated[
        str,
        typer.Option(
            help="Directory to write RF results (JSON + text report) to.",
        ),
    ] = "outputs/rf",
) -> None:
    """Run Random Forest feature importance analysis on the configured dataset variables."""
    result = run_rf_analysis(config)
    print_rf_table(result)

    json_path = save_rf_results(result, Path(output_dir))
    typer.echo(f"Results written to {json_path}")
