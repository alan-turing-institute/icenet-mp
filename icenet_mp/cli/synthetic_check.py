import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.synthetic import DYNAMICS_MOVING, run_synthetic_pipeline_check

from .hydra import hydra_adaptor

# Create the typer app
synthetic_cli = typer.Typer(help="Fast synthetic-data pipeline sanity checks")

log = logging.getLogger(__name__)


@synthetic_cli.command("check")
@hydra_adaptor
def check(  # noqa: PLR0913
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
    *,
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
    wandb: Annotated[
        bool,
        typer.Option(
            "--wandb",
            help=(
                "Also publish this run to W&B (entity 'turing-seaice'), alongside the "
                "local report files, under a random 'synthetic-' prefixed run name. "
                "Off by default: requires W&B credentials and publishes to a shared "
                "team workspace."
            ),
        ),
    ] = False,
    grid_size: Annotated[
        int,
        typer.Option(
            help=(
                "Height/width of the synthetic grid. The model's encoders.latent_space "
                "is set to match automatically. Must be > 16 and divisible by 16. "
                "Defaults to a small, fast 32; pass 432 to match real data's native "
                "resolution, at the cost of much longer training time."
            )
        ),
    ] = 32,
    n_trajectories: Annotated[
        int,
        typer.Option(
            help=(
                "Number of independent trajectories to generate (all but the last two "
                "go to training, one to validation, one to test). More trajectories "
                "means more/more-diverse data, at the cost of longer training time."
            )
        ),
    ] = 8,
    dynamics: Annotated[
        str,
        typer.Option(
            help=(
                "Which synthetic dynamics to generate: 'moving' (default) is a circle "
                "translating and bouncing off the grid edges; 'grow-shrink' is a "
                "stationary blob that grows and shrinks in place via a morphological "
                "open/close cycle, mimicking seasonal ice advance/retreat."
            )
        ),
    ] = DYNAMICS_MOVING,
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
        publish_wandb=wandb,
        grid_size=grid_size,
        n_trajectories=n_trajectories,
        dynamics=dynamics,
    )
    if result.wandb_run_name:
        log.info("Published to W&B as '%s'.", result.wandb_run_name)
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
