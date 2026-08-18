"""Optional, opt-in full-dataset debug videos for the synthetic pipeline check.

These are not part of the pass/fail gate -- they exist purely so a human can watch the
entire synthetic sequence, and the trained model's rollout predictions across the whole
thing, for debugging. Disabled by default since rendering and full-dataset inference
add real wall-clock time on top of the check itself.
"""

import datetime
import io
import logging
from pathlib import Path

import numpy as np
import torch

from icenet_mp.data import CombinedDataset, SingleDataset
from icenet_mp.models import BaseModel
from icenet_mp.types import ArrayTHW
from icenet_mp.utils import datetime_from_npdatetime
from icenet_mp.visualisations import DEFAULT_SIC_SPEC
from icenet_mp.visualisations.land_mask import LandMask
from icenet_mp.visualisations.plotting_video import (
    plot_video_prediction,
    plot_video_single_input,
)

logger = logging.getLogger(__name__)


def _write_first_video(videos: dict[str, io.BytesIO], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = next(iter(videos.values()))
    buffer.seek(0)
    output_path.write_bytes(buffer.read())


def write_full_dataset_video(
    *,
    frames: ArrayTHW,
    dates: list[datetime.datetime],
    variable_name: str,
    output_path: Path,
) -> None:
    """Render every generated frame as a single video, for visual sanity-checking."""
    buffer = plot_video_single_input(
        variable_name,
        frames,
        dates=dates,
        land_mask=LandMask(None),
        plot_spec=DEFAULT_SIC_SPEC,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer.seek(0)
    output_path.write_bytes(buffer.read())


def write_full_rollout_video(  # noqa: PLR0913
    *,
    model: BaseModel,
    zarr_path: Path,
    target_group_name: str,
    target_variables: list[str],
    n_history_steps: int,
    n_forecast_steps: int,
    variable_name: str,
    output_path: Path,
) -> None:
    """Roll the trained model out across every window in the dataset.

    Renders a ground-truth-vs-prediction video spanning the whole sequence (train,
    validation, and test windows together), unlike the evaluation plots which only
    sample a few test-split batches.
    """
    dataset = SingleDataset(target_group_name, [zarr_path])
    combined = CombinedDataset(
        [dataset],
        target_group_name=target_group_name,
        target_variables=target_variables,
        n_history_steps=n_history_steps,
        n_forecast_steps=n_forecast_steps,
    )

    model = model.to("cpu").eval()
    ground_truth_frames = []
    prediction_frames = []
    dates = []
    with torch.no_grad():
        for idx in range(len(combined)):
            batch = combined[idx]
            target = batch.pop("target")
            inputs = {
                name: torch.from_numpy(array).unsqueeze(0).float()
                for name, array in batch.items()
            }
            prediction = model(inputs)[0, 0, 0].numpy()

            ground_truth_frames.append(target[0, 0])
            prediction_frames.append(prediction)

            start_date = combined.dates[idx]
            forecast_date = combined.get_forecast_steps(start_date)[0]
            dates.append(datetime_from_npdatetime(forecast_date))

    ground_truth = np.stack(ground_truth_frames, axis=0)
    prediction = np.stack(prediction_frames, axis=0)

    logger.info(
        "Rendering full-dataset rollout video across %d windows (%s to %s).",
        len(dates),
        dates[0],
        dates[-1],
    )
    videos = plot_video_prediction(
        ground_truth,
        prediction,
        dates=dates,
        land_mask=LandMask(None),
        plot_spec=DEFAULT_SIC_SPEC,
        variable_name=variable_name,
    )
    _write_first_video(videos, output_path)
