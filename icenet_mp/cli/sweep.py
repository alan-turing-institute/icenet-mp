import logging
import os
from pathlib import Path
from typing import Annotated

import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from icenet_mp.model_service import ModelService
from icenet_mp.sweep import OptunaSampler

from .hydra import hydra_adaptor

# Create the typer app
sweep_cli = typer.Typer(help="Generate W&B sweeps with Optuna-sampled hyperparameters")

log = logging.getLogger(__name__)


@sweep_cli.command()
@hydra_adaptor
def initialise(
    config: DictConfig,
    sweep_yaml: Annotated[
        Path,
        typer.Option("--sweep-yaml", help="Path to a sweep search-space YAML file"),
    ],
) -> None:
    """Sample a batch of hyperparameter combinations and write a W&B sweep config.

    Writes `trials.json` (the sampled overrides for each trial) and `wandb_sweep.yaml`
    (a real W&B `method: grid` sweep over those trials) alongside the input file.
    """
    # Initialise Optuna sampler
    sampler = OptunaSampler.from_yaml(sweep_yaml)

    # Initialise a W&B sweep
    sweep_id = sampler.initialise_sweep(config)
    log.info("Initialised a W&B sweep with ID %s", sweep_id)

    sampler.initialise_study(config, sweep_id)
    log.info("Initialised an Optuna study at %s", sampler.study_path)


@sweep_cli.command()
def run(
    sweep_path: Annotated[
        Path,
        typer.Option(
            "--sweep-path",
            help="The path to the local sweep directory",
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
    # Load the Optuna sampler and start a trial
    sampler = OptunaSampler.from_path(sweep_path)
    trial = sampler.ask()

    # Generate parameter overrides and a merged config for this trial
    overrides = sampler.generate_parameter_overrides(trial)
    log.info("Running trial %d with overrides:", trial.number)
    for parameter, value in overrides:
        log.info("  %s = %s", parameter.name, value)
    config = sampler.generate_trial_config(overrides)

    # Set W&B environment variables to ensure we are connected to the correct sweep
    os.environ["WANDB_SWEEP_ID"] = sampler.study_name
    os.environ["WANDB_ENTITY"] = sampler.entity
    os.environ["WANDB_PROJECT"] = "train"

    # Train the model for this trial
    model = ModelService.from_config(config)
    trainer = model.train(
        checkpoint_dir=Path(checkpoint_dir).resolve() if checkpoint_dir else None,
        multistage=multistage,
    )

    # Record the trial result and log the best trial
    if not isinstance(ckpt := trainer.checkpoint_callback, ModelCheckpoint):
        return
    if ckpt.best_model_score is None:
        return
    result = sampler.tell(trial, ckpt.best_model_score.item())
    log.info("Trial %d completed with value %f", result.number, result.value)
    log.info(
        "Best trial (%d) completed with value %f.",
        sampler.study.best_trial.number,
        sampler.study.best_trial.value,
    )
    log.info("Best trial parameters: %s", sampler.study.best_trial.params)


if __name__ == "__main__":
    sweep_cli()
