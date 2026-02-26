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
        logger.info("Working on %s.", downloader.name)
        downloader.create(overwrite=overwrite)


@datasets_cli.command("inspect")
@hydra_adaptor
def inspect(config: DictConfig) -> None:
    """Inspect all datasets."""
    factory = DataDownloaderFactory(config)
    for downloader in factory.downloaders:
        logger.info("Working on %s.", downloader.name)
        downloader.inspect()


@datasets_cli.command("load")
@hydra_adaptor
def load(
    config: DictConfig,
    parts: Annotated[str, typer.Option(help="The part to process, specified as 'i/n'")],
) -> None:
    """Load dataset in parts."""
    factory = DataDownloaderFactory(config)
    for downloader in factory.downloaders:
        logger.info("Working on %s.", downloader.name)
        downloader.load(parts=parts)


@datasets_cli.command("finalise")
@hydra_adaptor
def finalise(
    config: DictConfig,
) -> None:
    """Finalise loaded dataset."""
    factory = DataDownloaderFactory(config)
    for downloader in factory.downloaders:
        logger.info("Working on %s.", downloader.name)
        downloader.finalise()


if __name__ == "__main__":
    datasets_cli()
