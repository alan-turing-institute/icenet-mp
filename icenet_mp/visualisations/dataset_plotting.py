from dataclasses import replace
from pathlib import Path

from icenet_mp.data import SingleDataset
from icenet_mp.utils import datetime_from_npdatetime

from .helpers import DEFAULT_SIC_SPEC
from .land_mask import LandMask
from .plotting_static import plot_static_inputs
from .plotting_video import plot_video_inputs


def plot_variables_static(
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
    plot_spec = replace(DEFAULT_SIC_SPEC, hemisphere=dataset.hemisphere)
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


def plot_variables_video(
    dataset_name: str,
    dataset_path: Path,
    output_dir: Path,
    timestep: int,
    n_steps: int,
) -> int:
    """Save one animation per variable for a run of consecutive timesteps."""
    dataset = SingleDataset(
        name=dataset_name,
        input_files=[dataset_path],
        normalise=False,
    )
    plot_spec = replace(DEFAULT_SIC_SPEC, hemisphere=dataset.hemisphere)
    if timestep < 0 or n_steps < 1 or timestep + n_steps > len(dataset):
        msg = (
            f"Timesteps {timestep}:{timestep + n_steps} are out of range for dataset "
            f"{dataset_name} with {len(dataset)} timesteps"
        )
        raise IndexError(msg)

    dates = [
        datetime_from_npdatetime(date)
        for date in dataset.dates[timestep : timestep + n_steps]
    ]
    tchw = dataset.get_tchw_slice(dataset.dates[timestep], n_steps)
    variables = {
        f"{dataset.name}:{variable_name}": tchw[:, channel]
        for channel, variable_name in enumerate(dataset.variable_names)
    }
    videos = plot_video_inputs(
        variables,
        dates=dates,
        land_mask=LandMask(None),
        plot_spec=plot_spec,
    )

    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for video_name, video_buffer in videos.items():
        safe_name = video_name.replace(":", "_").replace("/", "_")
        video_buffer.seek(0)
        video_path = dataset_output_dir / f"{safe_name}.{plot_spec.video_format}"
        video_path.write_bytes(video_buffer.read())
        saved += 1
    return saved
