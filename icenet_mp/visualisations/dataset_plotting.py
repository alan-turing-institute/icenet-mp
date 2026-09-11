from dataclasses import replace
from pathlib import Path

from icenet_mp.data import SingleDataset
from icenet_mp.utils import datetime_from_npdatetime, mask_dir

from .default_plot_spec import DEFAULT_SIC_SPEC
from .land_mask import LandMask
from .panel_renderer import PanelRenderer


def plot_variables_static(
    *,
    base_path: Path,
    dataset_name: str,
    dataset_path: Path,
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
    land_mask_path = mask_dir(base_path, dataset_name) / "land_mask.npy"
    renderer = PanelRenderer(LandMask(land_mask_path), plot_spec)

    dataset_output_dir = base_path / "data" / "input_plots" / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for variable_name, variable_values in variables.items():
        image = renderer.static_singlet(
            variable_values,
            when=when,
            variable_name=variable_name,
        )
        image_name = f"{when.strftime(r'%Y-%m-%d')}-{variable_name}"
        safe_name = image_name.replace(":", "_").replace("/", "_")
        image.save(dataset_output_dir / f"{safe_name}.png")
        saved += 1
    return saved


def plot_variables_video(
    *,
    base_path: Path,
    dataset_name: str,
    dataset_path: Path,
    n_steps: int,
    timestep: int,
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
    land_mask_path = mask_dir(base_path, dataset_name) / "land_mask.npy"
    renderer = PanelRenderer(LandMask(land_mask_path), plot_spec)

    dataset_output_dir = base_path / "data" / "input_plots" / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for variable_name, variable_values in variables.items():
        video_buffer = renderer.video_singlet(
            variable_values,
            dates=dates,
            variable_name=variable_name,
        )
        video_name = f"{dates[0].strftime(r'%Y-%m-%d')}-{variable_name}"
        safe_name = video_name.replace(":", "_").replace("/", "_")
        video_buffer.seek(0)
        video_path = dataset_output_dir / f"{safe_name}.{plot_spec.video_format}"
        video_path.write_bytes(video_buffer.read())
        saved += 1
    return saved
