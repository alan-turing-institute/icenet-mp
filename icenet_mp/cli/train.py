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
def pretrain(
    config: DictConfig,
    checkpoint: Annotated[
        str | None,
        typer.Option(
            help=(
                "Path to a directory of existing encoder checkpoints. "
                "Any encoder whose checkpoint ({name}.ckpt) already exists there will skip training."
            )
        ),
    ] = None,
) -> None:
    """Pretrain an autoencoder model."""
    model = ModelService.from_config(config)
    model.pretrain(checkpoint_dir=Path(checkpoint).resolve() if checkpoint else None)


@training_cli.command()
@hydra_adaptor
def train(config: DictConfig) -> None:
    """Train a model."""
    model = ModelService.from_config(config)
    model.train()


if __name__ == "__main__":
    training_cli()
