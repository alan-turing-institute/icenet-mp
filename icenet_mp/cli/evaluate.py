import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.model_service import ModelService
from icenet_mp.visualisations.hook_manager import ActivationHookManager

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
    save_activations: Annotated[
        bool,
        typer.Option(
            "--save-activations/--no-save-activations",
            help=(
                "Register forward hooks on selected model layers during the "
                "test loop and save captured activations to disk."
            ),
        ),
    ] = False,
    activations_output: Annotated[
        Path | None,
        typer.Option(
            "--activations-output",
            help=(
                "Directory to write per-batch activation files. "
                "Required when --save-activations is set."
            ),
        ),
    ] = None,
    activation_layer: Annotated[
        list[str] | None,
        typer.Option(
            "--activation-layer",
            help=(
                "Dotted path of a model submodule to hook (e.g. 'processor.conv1'). "
                "Repeat the flag to hook multiple layers."
            ),
        ),
    ] = None,
) -> None:
    """Evaluate a pre-trained model."""
    model = ModelService.from_checkpoint(config, Path(checkpoint).resolve())

    if save_activations:
        if activations_output is None:
            msg = "--activations-output is required when --save-activations is set."
            raise typer.BadParameter(msg)
        layers = list(activation_layer or [])
        if not layers:
            msg = (
                "At least one --activation-layer must be specified when "
                "--save-activations is set."
            )
            raise typer.BadParameter(msg)

        hook_manager = ActivationHookManager(
            model=model.model,  # maybe slightly confusing naming bere
            layer_paths=layers,
            output_dir=activations_output,
        )
        hook_manager.attach()
        # Append directly: ModelService.add_callbacks expects DictConfig entries
        # to hydra-instantiate, whereas this callback is already built.
        model.extra_callbacks_.append(hook_manager)
        log.info(
            "Activation capture enabled for layers %s; writing to %s",
            layers,
            activations_output,
        )

    model.evaluate()


if __name__ == "__main__":
    evaluation_cli()
