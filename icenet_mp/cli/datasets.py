import logging
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig

from icenet_mp.data import SingleDataset
from icenet_mp.ingestion import build_downloaders
from icenet_mp.ingestion.data_downloader import DataDownloader
from icenet_mp.utils import datetime_from_npdatetime
from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.plotting_static import plot_static_inputs

from .hydra import hydra_adaptor

# Create the typer app
datasets_cli = typer.Typer(help="Manage datasets")

logger = logging.getLogger(__name__)


def _plot_downloader(
    downloader: DataDownloader,
    output_dir: Path,
    timestep: int,
) -> int:
    """Save static plots for one timestep of a downloaded dataset."""
    if not downloader.path_dataset.exists():
        msg = f"Dataset {downloader.name} not found at {downloader.path_dataset}"
        raise RuntimeError(msg)

    dataset = SingleDataset(
        name=downloader.name,
        input_files=[downloader.path_dataset],
        normalise=False,
    )
    if timestep < 0 or timestep >= len(dataset):
        msg = (
            f"Timestep {timestep} is out of range for dataset {downloader.name} "
            f"with {len(dataset)} timesteps"
        )
        raise IndexError(msg)

    when = datetime_from_npdatetime(dataset.dates[timestep])
    variables = {
        f"{dataset.name}:{variable_name}": dataset[timestep][channel]
        for channel, variable_name in enumerate(dataset.variable_names)
    }
    plot_spec = replace(DEFAULT_SIC_SPEC, hemisphere=dataset.hemisphere)
    images = plot_static_inputs(
        variables,
        land_mask=LandMask(None),
        plot_spec=plot_spec,
        when=when,
    )

    dataset_output_dir = output_dir / downloader.name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for image_name, image_list in images.items():
        safe_name = image_name.replace(":", "_").replace("/", "_")
        for idx_image, image in enumerate(image_list):
            suffix = f"-{idx_image}" if len(image_list) > 1 else ""
            image.save(dataset_output_dir / f"{safe_name}{suffix}.png")
            saved += 1
    return saved


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
    output_dir: Annotated[
        Path, typer.Option(help="Directory in which to save dataset plots")
    ] = Path("dataset_plots"),
    timestep: Annotated[int, typer.Option(help="Dataset timestep index to plot")] = 0,
) -> None:
    """Plot one timestep of configured datasets."""
    matched_dataset = False
    for downloader in build_downloaders(config):
        if dataset is not None and downloader.name != dataset:
            continue
        matched_dataset = True
        logger.info("Plotting dataset %s.", downloader.name)
        try:
            n_saved = _plot_downloader(downloader, output_dir, timestep)
        except (IndexError, RuntimeError, ValueError, OSError) as exc:
            logger.error(  # noqa: TRY400
                "Plotting dataset %s failed, skipping: %s", downloader.name, exc
            )
            continue
        logger.info(
            "Saved %d plots for dataset %s under %s.",
            n_saved,
            downloader.name,
            output_dir / downloader.name,
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
