import json
import logging
import os
from pathlib import Path
from typing import Annotated

import typer
import wandb
import yaml

from icenet_mp.cli.train import train as train_cli
from icenet_mp.sweep import OptunaSampler

# Create the typer app
sweep_cli = typer.Typer(help="Generate W&B sweeps with Optuna-sampled hyperparameters")

log = logging.getLogger(__name__)


@sweep_cli.command()
def generate(
    sweep_yaml: Annotated[
        Path,
        typer.Argument(help="Path to a sweep search-space YAML file"),
    ],
    entity: Annotated[
        str | None,
        typer.Option(
            "--entity",
            help=("The W&B entity (default: WANDB_ENTITY environment variable)"),
        ),
    ] = None,
) -> None:
    """Sample a batch of hyperparameter combinations and write a W&B sweep config.

    Writes `trials.json` (the sampled overrides for each trial) and `wandb_sweep.yaml`
    (a real W&B `method: grid` sweep over those trials) alongside the input file.
    """
    sampler = OptunaSampler.from_yaml(sweep_yaml)

    # Generate trials
    trials = sampler.generate_trials()
    trials_path = sweep_yaml.parent / "trials.json"
    trials_path.write_text(json.dumps(trials, indent=2))
    log.info("Wrote %d sampled combinations to %s", len(trials), trials_path)

    # Generate W&B sweep config
    sweep_config = sampler.generate_sweep_config(trials)
    sweep_config_path = sweep_yaml.parent / "wandb_sweep.yaml"
    sweep_config_path.write_text(yaml.safe_dump(sweep_config, sort_keys=False))
    log.info("Wrote W&B sweep config to %s", sweep_config_path)

    # Initialise W&B sweep
    entity = entity or os.environ.get("WANDB_ENTITY")
    sweep_id = wandb.sweep(sweep_config, entity=entity, project="train")
    log.info("Initialised sweep with id %s", sweep_id)


@sweep_cli.command()
def run(
    trial_number: Annotated[
        int,
        typer.Option(
            "--trial-number",
            help=("The trial number to run (0-indexed)."),
        ),
    ],
    sweep_id: Annotated[
        str | None,
        typer.Option(
            "--sweep-id",
            help=("The W&B sweep ID (default: WANDB_SWEEP_ID environment variable)"),
        ),
    ] = None,
    config_name: Annotated[
        str | None,
        typer.Option(
            help="Specify the name of a file to load from the config directory"
        ),
    ] = "sample",
    entity: Annotated[
        str | None,
        typer.Option(
            "--entity",
            help=("The W&B entity (default: WANDB_ENTITY environment variable)"),
        ),
    ] = None,
) -> None:
    """Run a single trial from a W&B sweep."""
    sweep_id = sweep_id or os.environ.get("WANDB_SWEEP_ID")
    if sweep_id is None:
        msg = "No sweep ID given. Pass --sweep-id or set WANDB_SWEEP_ID."
        raise typer.BadParameter(msg)

    trials_path = Path("trials.json")
    if not trials_path.exists():
        msg = f"Missing {trials_path}, did you run `imp sweep generate`?"
        raise typer.BadParameter(msg)

    trials = json.loads(trials_path.read_text())
    if trial_number < 0 or trial_number >= len(trials):
        msg = f"Trial number {trial_number} is out of range (0-{len(trials) - 1})"
        raise typer.BadParameter(msg)

    overrides = [f"++{key}={value}" for key, value in trials[trial_number].items()]
    overrides.append(f"+sweep_trial_number={trial_number}")

    # Set W&B environment variables to ensure we are connected to the correct sweep
    os.environ["WANDB_SWEEP_ID"] = sweep_id
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_PROJECT"] = "train"

    # Start running the trial
    log.info("Running trial %d with overrides: %s", trial_number, overrides)
    train_cli(overrides=overrides, config_name=config_name)  # type: ignore[arg-type]


if __name__ == "__main__":
    sweep_cli()
