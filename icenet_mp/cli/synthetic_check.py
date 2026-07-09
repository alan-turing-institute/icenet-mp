import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.synthetic import run_synthetic_pipeline_check

from .hydra import hydra_adaptor

# Create the typer app
synthetic_cli = typer.Typer(help="Fast synthetic-data pipeline sanity checks")

log = logging.getLogger(__name__)


@synthetic_cli.command("check")
@hydra_adaptor
def check(
    config: DictConfig,
    output_dir: Annotated[
        str,
        typer.Option(
            help=(
                "Directory to write the generated synthetic dataset, checkpoints, "
                "loss-curve/prediction plots, and pass/fail report to."
            )
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
    debug_full_video: Annotated[
        bool,
        typer.Option(
            "--debug-full-video",
            help=(
                "Also render the entire generated dataset, and the trained model's "
                "rollout across the whole dataset (not just sampled evaluation "
                "batches), as videos under output_dir/report/debug. Off by default: "
                "this re-runs inference across every window and adds real time."
            ),
        ),
    ] = False,
) -> None:
    """Run a fast synthetic moving-circle pipeline sanity check.

    Trains and evaluates the model+data configuration given by `config_name`/`overrides`
    (exactly as for a real job) against a small, deterministic moving-circle dataset,
    then asserts that the model actually learned. Loss-curve and
    ground-truth-vs-prediction plots are written to `output_dir` for local inspection,
    and the command exits non-zero on failure so it can gate CI.
    """
    result = run_synthetic_pipeline_check(
        config,
        output_dir=Path(output_dir),
        max_epochs=max_epochs,
        min_relative_improvement=min_relative_improvement,
        dump_debug_video=debug_full_video,
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
    synthetic_cli()
