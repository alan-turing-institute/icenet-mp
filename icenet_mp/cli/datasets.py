import logging
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.ingestion import build_downloaders
from icenet_mp.visualisations import plot_variables_static, plot_variables_video

from .hydra import hydra_adaptor

# Create the typer app
datasets_cli = typer.Typer(help="Manage datasets")

logger = logging.getLogger(__name__)


@datasets_cli.command("create")
@hydra_adaptor
def create(
    config: DictConfig,
    *,
    overwrite: Annotated[
        bool, typer.Option(help="Specify whether to overwrite existing datasets")
    ] = False,
) -> None:
    """Create all datasets."""
    for downloader in build_downloaders(config):
        logger.info("Creating dataset %s.", downloader.name)
        try:
            downloader.create(overwrite=overwrite)
        except RuntimeError as exc:
            logger.error("Failed to create %s: %s", downloader.name, exc)  # noqa: TRY400
            raise typer.Exit(1) from exc


@datasets_cli.command("inspect")
@hydra_adaptor
def inspect(
    config: DictConfig,
    *,
    verbose: Annotated[
        bool, typer.Option(help="Show detailed dataset information")
    ] = False,
) -> None:
    """Inspect all datasets."""
    for downloader in build_downloaders(config):
        logger.info("Inspecting dataset %s.", downloader.name)
        try:
            downloader.inspect(verbose=verbose)
        except RuntimeError:
            logger.error("Inspecting dataset %s failed, skipping.", downloader.name)  # noqa: TRY400


@datasets_cli.command("plot")
@hydra_adaptor
def plot(
    config: DictConfig,
    *,
    dataset: Annotated[
        str | None, typer.Option(help="Only plot the named configured dataset")
    ] = None,
    timestep: Annotated[int, typer.Option(help="Dataset timestep index to plot")] = 0,
    video: Annotated[
        bool,
        typer.Option(
            help="Animate --n-steps consecutive timesteps instead of plotting one"
        ),
    ] = False,
    n_steps: Annotated[
        int,
        typer.Option(help="Number of consecutive timesteps to animate with --video"),
    ] = 10,
) -> None:
    """Plot one timestep of configured datasets."""
    base_path = Path(config["base_path"]).resolve()
    matched_dataset = False
    for downloader in build_downloaders(config):
        if dataset is not None and downloader.name != dataset:
            continue
        matched_dataset = True
        logger.info("Plotting dataset %s.", downloader.name)
        if downloader.path_dataset.exists():
            n_saved = (
                plot_variables_video(
                    downloader.name,
                    downloader.path_dataset,
                    timestep,
                    n_steps,
                    base_path=base_path,
                )
                if video
                else plot_variables_static(
                    downloader.name,
                    downloader.path_dataset,
                    timestep,
                    base_path=base_path,
                )
            )
            logger.info("Saved %d plots for dataset %s.", n_saved, downloader.name)
        else:
            logger.error(
                "Dataset %s not found at %s", downloader.name, downloader.path_dataset
            )
    if dataset is not None and not matched_dataset:
        logger.error("Configured dataset %s was not found.", dataset)
        raise typer.Exit(1)


@datasets_cli.command("masks")
@hydra_adaptor
def masks(
    config: DictConfig,
    *,
    overwrite: Annotated[
        bool, typer.Option(help="Specify whether to overwrite existing masks")
    ] = False,
) -> None:
    """Create land / active grid cell masks."""
    for downloader in build_downloaders(config):
        logger.info("Generating masks for dataset %s.", downloader.name)
        downloader.postprocessor.process(downloader.path_dataset, overwrite=overwrite)


if __name__ == "__main__":
    datasets_cli()
