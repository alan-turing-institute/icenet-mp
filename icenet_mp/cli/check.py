import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.synthetic.pipeline_check import run_synthetic_pipeline_check

from .hydra import hydra_adaptor

check_cli = typer.Typer(help="Run pipeline checks")

log = logging.getLogger(__name__)


@check_cli.command()
@hydra_adaptor
def check(
    config: DictConfig,
    output_dir: Annotated[
        str,
        typer.Option(
            help="Directory receiving checkpoints, reports, and other check outputs."
        ),
    ] = "outputs/synthetic_check",
    max_epochs: Annotated[
        int | None,
        typer.Option(help="Override train.trainer.max_epochs for this run."),
    ] = None,
    min_relative_improvement: Annotated[
        float,
        typer.Option(
            help=(
                "Minimum fractional drop from the first epoch's validation loss to "
                "the best epoch's, required for the check to pass."
            )
        ),
    ] = 0.3,
) -> None:
    """Run a fast synthetic-data pipeline sanity check."""
    result = run_synthetic_pipeline_check(
        config,
        output_dir=Path(output_dir),
        max_epochs=max_epochs,
        min_relative_improvement=min_relative_improvement,
    )
    if result.passed:
        log.info("Synthetic pipeline check PASSED. Report: %s", result.report_path)
    else:
        log.error(
            "Synthetic pipeline check FAILED: %s Report: %s",
            " ".join(result.reasons),
            result.report_path,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    check_cli()
