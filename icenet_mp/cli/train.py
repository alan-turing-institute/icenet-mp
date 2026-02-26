import logging

import typer
from omegaconf import DictConfig

from icenet_mp.model_service import ModelService

from .hydra import hydra_adaptor

# Create the typer app
training_cli = typer.Typer(help="Train models")

log = logging.getLogger(__name__)


@training_cli.command()
@hydra_adaptor
def train(config: DictConfig) -> None:
    """Train a model."""
    model = ModelService.from_config(config)
    model.train()


if __name__ == "__main__":
    training_cli()
