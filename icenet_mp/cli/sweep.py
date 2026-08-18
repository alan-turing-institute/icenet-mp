import logging
import os
from pathlib import Path
from typing import Annotated

import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from optuna.trial import TrialState

from icenet_mp.model_service import ModelService
from icenet_mp.sweep import OptunaSweep

from .hydra import hydra_adaptor

# Configure logging
log = logging.getLogger(__name__)

# Create the typer app
sweep_cli = typer.Typer(help="Generate W&B sweeps with Optuna-sampled hyperparameters")


@sweep_cli.command()
@hydra_adaptor
def initialise(
    config: DictConfig,
    sweep_yaml: Annotated[
        Path,
        typer.Option(
            "--sweep-yaml", help="Full path to a sweep search-space YAML file"
        ),
    ],
) -> None:
    """Initialise a W&B sweep with Optuna-sampled hyperparameters.

    Create a W&B sweep with a grid search over a set of hyperparameter combinations
    specified by Optuna. Create the Optuna study directory and save the model and sweep
    configs.
    """
    sweep = OptunaSweep.from_yaml(sweep_yaml)

    # Initialise a W&B sweep
    sweep_id = sweep.initialise_sweep(config)
    log.info("Initialised a W&B sweep with ID %s", sweep_id)

    # Initialise Optuna study
    sweep.initialise_study(config, sweep_id)
    log.info("Initialised an Optuna study at %s", sweep.study_path)


@sweep_cli.command()
def summarise(
    sweep_path: Annotated[
        Path,
        typer.Option(
            "--sweep-path",
            help="Full path to a local sweep directory",
        ),
    ],
) -> None:
    """Summarise the best parameters found in a W&B sweep."""
    sweep = OptunaSweep.from_path(sweep_path)
    trials = sweep.study.get_trials()
    log.info("Study contains %d trial(s)", len(trials))
    if not any(t.state == TrialState.COMPLETE for t in trials):
        log.info("No trials have completed yet. Nothing to summarise.")
        return
    best = sweep.study.best_trial
    log.info("Best trial (%d) completed with value %f.", best.number, best.value)
    log.info("Best trial parameters:")
    for parameter_name, parameter_value in best.params.items():
        log.info("  %s: %s", parameter_name, parameter_value)


@sweep_cli.command()
def trial(
    sweep_path: Annotated[
        Path,
        typer.Option(
            "--sweep-path",
            help="Full path to a local sweep directory",
        ),
    ],
    *,
    checkpoint_dir: Annotated[
        str | None,
        typer.Option(
            "--checkpoint-dir",
            help=(
                "Path to a directory of existing checkpoints. If a checkpoint exists "
                "in this directory for any model component, it will be loaded and "
                "training will be skipped for that component. "
            ),
        ),
    ] = None,
    multistage: Annotated[
        bool,
        typer.Option(
            "--multistage",
            help=(
                "Train an EncodeProcessDecode model in multiple stages (encoders, then "
                "decoder, then processor, then finetune). Default is single-stage "
                "training."
            ),
        ),
    ] = False,
) -> None:
    """Run a single trial from a W&B sweep."""
    # Load the Optuna sweep, start a trial, and get its parameter overrides
    sweep = OptunaSweep.from_path(sweep_path)
    trial, overrides = sweep.ask()

    # Generate a merged config for this trial
    log.info("Running trial %d with overrides:", trial.number)
    for parameter, value in overrides:
        log.info("  %s = %s", parameter.name, value)
    config = sweep.generate_trial_config(overrides)

    # Set W&B environment variables to ensure we are connected to the correct sweep
    os.environ["WANDB_SWEEP_ID"] = sweep.study_name
    os.environ["WANDB_ENTITY"] = sweep.entity
    os.environ["WANDB_PROJECT"] = "train"

    # Train the model for this trial
    model = ModelService.from_config(config)
    trainer = model.train(
        checkpoint_dir=Path(checkpoint_dir).resolve() if checkpoint_dir else None,
        multistage=multistage,
    )

    # Find the checkpoint callback to score the trial against
    checkpoints = [
        ckpt
        for ckpt in trainer.checkpoint_callbacks
        if isinstance(ckpt, ModelCheckpoint)
    ]
    if len(checkpoints) > 1:
        msg = (
            f"Cannot determine which of {len(checkpoints)} checkpoint callbacks to "
            "score the trial against."
        )
        raise ValueError(msg)
    checkpoint = checkpoints[0] if checkpoints else None

    # Record the trial result and log the best trial
    if checkpoint is None:
        log.warning(
            "Trial %d failed: no ModelCheckpoint callback was configured, so it "
            "cannot be scored.",
            trial.number,
        )
        sweep.tell(trial, state=TrialState.FAIL)
        return
    if checkpoint.best_model_score is None:
        log.warning(
            "Trial %d failed: the checkpoint callback monitoring '%s' has no "
            "best_model_score, so it cannot be scored.",
            trial.number,
            checkpoint.monitor,
        )
        sweep.tell(trial, state=TrialState.FAIL)
        return
    result = sweep.tell(trial, checkpoint.best_model_score.item())
    log.info("Trial %d completed with value %f", result.number, result.value)
    best = sweep.study.best_trial
    log.info("Best trial (%d) completed with value %f.", best.number, best.value)
    log.info("Best trial parameters:")
    for parameter_name, parameter_value in best.params.items():
        log.info("  %s: %s", parameter_name, parameter_value)


if __name__ == "__main__":
    sweep_cli()
