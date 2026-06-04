import logging
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.data_processors import DataDownloaderFactory

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
    factory = DataDownloaderFactory(config)
    for downloader in factory.downloaders:
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
    statistics: Annotated[
        bool, typer.Option(help="Recalculate dataset statistics")
    ] = False,
    verbose: Annotated[
        bool, typer.Option(help="Show detailed dataset information")
    ] = False,
) -> None:
    """Inspect all datasets."""
    factory = DataDownloaderFactory(config)
    for downloader in factory.downloaders:
        logger.info("Inspecting dataset %s.", downloader.name)
        try:
            downloader.inspect(statistics=statistics, verbose=verbose)
        except RuntimeError:
            logger.error("Inspecting dataset %s failed, skipping.", downloader.name)  # noqa: TRY400


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
    factory = DataDownloaderFactory(config)
    for downloader in factory.downloaders:
        logger.info("Generating masks for dataset %s.", downloader.name)
        downloader.generate_masks(overwrite=overwrite)


if __name__ == "__main__":
    datasets_cli()
