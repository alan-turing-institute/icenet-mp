import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.model_service import ModelService

from .hydra import hydra_adaptor

# Create the typer app
training_cli = typer.Typer(help="Train models")

log = logging.getLogger(__name__)


@training_cli.command()
@hydra_adaptor
def train(
    config: DictConfig,
    checkpoint_dir: Annotated[
        str | None,
        typer.Option(
            help=(
                "Path to a directory of existing checkpoints. If a checkpoint exists "
                "in this directory for any model component, it will be loaded and "
                "training will be skipped for that component. "
            )
        ),
    ] = None,
    *,
    in_stages: Annotated[
        bool,
        typer.Option(
            "--in-stages",
            help=(
                "Train a composable model in stages (encoders, then decoder, "
                "then processor, then finetune). Default is end-to-end training."
            ),
        ),
    ] = False,
) -> None:
    """Train a model."""
    model = ModelService.from_config(config)
    model.train(
        checkpoint_dir=Path(checkpoint_dir).resolve() if checkpoint_dir else None,
        in_stages=in_stages,
    )


if __name__ == "__main__":
    training_cli()
