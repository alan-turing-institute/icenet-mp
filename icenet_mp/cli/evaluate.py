import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.model_service import ModelService

from .hydra import hydra_adaptor

# Create the typer app
evaluation_cli = typer.Typer(help="Evaluate models")

log = logging.getLogger(__name__)


@evaluation_cli.command()
@hydra_adaptor
def evaluate(
    config: DictConfig,
    checkpoint: Annotated[
        str, typer.Option(help="Specify the path to a trained model checkpoint")
    ],
    save_layer: Annotated[
        list[str] | None,
        typer.Option(
            "--save-layer",
            help=(
                "Dotted path of a model submodule to hook (e.g. 'processor.conv1'). "
                "Repeat the flag to hook multiple layers. "
                "Values for each selected layer will be saved to disk each batch."
            ),
        ),
    ] = None,
) -> None:
    """Evaluate a pre-trained model."""
    # If activation saving is enabled, then add requested layers
    if layer_paths := list(save_layer or []):
        config.get("evaluate", {}).get("callbacks", {}).get("activation_saver", {})[
            "layer_paths"
        ] = layer_paths
    model = ModelService.from_checkpoint(config, Path(checkpoint).resolve())
    model.evaluate()


if __name__ == "__main__":
    evaluation_cli()
