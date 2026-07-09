"""Compose several independent moving-circle trajectories into one dataset.

A single long trajectory sliced into overlapping windows gives a model many
near-identical views of one physical instance, not genuine examples to generalise
from. Concatenating several independent trajectories -- each with a different start
position and velocity, separated by gap days so no window can bridge two of them --
lets whole trajectories be held out for validation/test, so the check actually
exercises whether a model learned the general translate-and-bounce rule rather than
memorising one specific path. The per-trajectory task stays exactly as simple.
"""

import datetime
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from icenet_mp.types import ArrayTHW

from .shapes import MovingCircleConfig, generate_moving_circle_frames
from .zarr_writer import daily_dates


@dataclass(frozen=True)
class TrajectorySpan:
    """The calendar date range (inclusive) occupied by one trajectory."""

    start_date: datetime.datetime
    end_date: datetime.datetime


@dataclass(frozen=True)
class MultiTrajectoryDataset:
    frames: ArrayTHW
    dates: list[datetime.datetime]
    missing_dates: list[datetime.datetime]
    spans: list[TrajectorySpan]


def default_trajectory_configs(
    *,
    height: int = 32,
    width: int = 32,
    n_timesteps: int = 20,
    radius: float | None = None,
    n_trajectories: int = 8,
) -> tuple[MovingCircleConfig, ...]:
    """`n_trajectories` bouncing-circle trajectories with distinct start/velocity.

    Start positions are spread evenly around a ring centred on the grid; velocity
    directions are roughly tangential, alternating a +45-degree offset every other
    trajectory, with two alternating speeds. Varying the (start position, velocity)
    combination -- while keeping the circle, grid, and update rule identical -- means
    a model must learn the general translate-and-bounce rule rather than memorise one
    specific path. `radius`/velocity magnitude scale with grid size (relative to a
    32x32 reference) unless `radius` is given explicitly, so the task looks the same
    at any resolution.
    """
    scale = height / 32
    radius = radius if radius is not None else 5.0 * scale
    centre_row, centre_col = height / 2, width / 2
    start_radius = min(height, width) / 2 - radius - 2 * scale
    speeds = (2.0 * scale, 3.0 * scale)

    trajectories = []
    for i in range(n_trajectories):
        angle = 2 * np.pi * i / n_trajectories
        start_position = (
            centre_row + start_radius * np.sin(angle),
            centre_col + start_radius * np.cos(angle),
        )
        speed = speeds[i % len(speeds)]
        velocity_angle = angle + np.pi / 2 + (np.pi / 4 if i % 2 else 0.0)
        velocity = (speed * np.sin(velocity_angle), speed * np.cos(velocity_angle))
        trajectories.append(
            MovingCircleConfig(
                height=height,
                width=width,
                n_timesteps=n_timesteps,
                radius=radius,
                start_position=start_position,
                velocity=velocity,
            )
        )
    return tuple(trajectories)


def generate_multi_trajectory_dataset(
    trajectories: Sequence[MovingCircleConfig],
    *,
    gap_days: int = 2,
    start_date: datetime.datetime = datetime.datetime(2020, 1, 1),  # noqa: B008
) -> MultiTrajectoryDataset:
    """Concatenate independent trajectories end-to-end, with gap days between them."""
    frame_chunks: list[ArrayTHW] = []
    spans: list[TrajectorySpan] = []
    missing_dates: list[datetime.datetime] = []

    n_total = sum(t.n_timesteps for t in trajectories) + gap_days * (
        len(trajectories) - 1
    )
    all_dates = daily_dates(n_total, start_date)

    cursor = 0
    for index, traj_config in enumerate(trajectories):
        frame_chunks.append(generate_moving_circle_frames(traj_config))
        spans.append(
            TrajectorySpan(
                start_date=all_dates[cursor],
                end_date=all_dates[cursor + traj_config.n_timesteps - 1],
            )
        )
        cursor += traj_config.n_timesteps

        if index < len(trajectories) - 1:
            missing_dates.extend(all_dates[cursor : cursor + gap_days])
            frame_chunks.append(
                np.zeros(
                    (gap_days, traj_config.height, traj_config.width), dtype=np.float32
                )
            )
            cursor += gap_days

    return MultiTrajectoryDataset(
        frames=np.concatenate(frame_chunks, axis=0),
        dates=all_dates,
        missing_dates=missing_dates,
        spans=spans,
    )
