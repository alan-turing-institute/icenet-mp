from dataclasses import replace
from pathlib import Path

from icenet_mp.data import SingleDataset
from icenet_mp.utils import datetime_from_npdatetime

from .helpers import DEFAULT_SIC_SPEC
from .land_mask import LandMask
from .plotting_static import plot_static_inputs


def plot_dataset(
    dataset_name: str,
    dataset_path: Path,
    output_dir: Path,
    timestep: int,
) -> int:
    """Save static plots for one timestep of a downloaded dataset."""
    dataset = SingleDataset(
        name=dataset_name,
        input_files=[dataset_path],
        normalise=False,
    )
    if timestep < 0 or timestep >= len(dataset):
        msg = (
            f"Timestep {timestep} is out of range for dataset {dataset_name} "
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

    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for image_name, image_list in images.items():
        safe_name = image_name.replace(":", "_").replace("/", "_")
        for idx_image, image in enumerate(image_list):
            suffix = f"-{idx_image}" if len(image_list) > 1 else ""
            image.save(dataset_output_dir / f"{safe_name}{suffix}.png")
            saved += 1
    return saved
