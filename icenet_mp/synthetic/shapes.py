"""Deterministic synthetic spatiotemporal sequences for fast model-pipeline checks.

A moving circle on a periodic (torus) grid gives a trivially-learnable spatiotemporal
forecasting task with a known ground truth, so it can validate that a model+data
pipeline is wired up correctly (shapes, history/forecast windowing, rollout) and that a
model is actually learning, without needing real sea-ice data.
"""

from dataclasses import dataclass

import numpy as np

from icenet_mp.types import ArrayTHW


@dataclass(frozen=True)
class MovingCircleConfig:
    """Configuration for a single circle translating across a periodic HxW grid."""

    height: int = 32
    width: int = 32
    n_timesteps: int = 60
    radius: float = 6.0
    velocity: tuple[float, float] = (2.0, 1.0)
    start_position: tuple[float, float] | None = None
    foreground_value: float = 1.0
    background_value: float = 0.0


def generate_moving_circle_frames(config: MovingCircleConfig) -> ArrayTHW:
    """Generate a [T, H, W] sequence of a circle translating across a periodic grid.

    The circle wraps around the grid edges (torus topology), so every frame contains a
    fully-formed circle regardless of position, with no boundary artefacts.
    """
    height, width = config.height, config.width
    start_row, start_col = config.start_position or (height / 2, width / 2)
    v_row, v_col = config.velocity

    row_grid, col_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    frames = np.empty((config.n_timesteps, height, width), dtype=np.float32)
    for t in range(config.n_timesteps):
        centre_row = (start_row + v_row * t) % height
        centre_col = (start_col + v_col * t) % width
        d_row = np.minimum(
            np.abs(row_grid - centre_row), height - np.abs(row_grid - centre_row)
        )
        d_col = np.minimum(
            np.abs(col_grid - centre_col), width - np.abs(col_grid - centre_col)
        )
        inside = (d_row**2 + d_col**2) <= config.radius**2
        frames[t] = np.where(inside, config.foreground_value, config.background_value)

    return frames
