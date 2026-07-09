"""Deterministic synthetic spatiotemporal sequences for fast model-pipeline checks.

A moving circle gives a trivially-learnable spatiotemporal forecasting task with a
known ground truth, so it can validate that a model+data pipeline is wired up correctly
(shapes, history/forecast windowing, rollout) and that a model is actually learning,
without needing real sea-ice data.
"""

from dataclasses import dataclass

import numpy as np

from icenet_mp.types import ArrayTHW


@dataclass(frozen=True)
class MovingCircleConfig:
    """Configuration for a single circle bouncing around inside an HxW grid."""

    height: int = 32
    width: int = 32
    n_timesteps: int = 60
    radius: float = 5.0
    velocity: tuple[float, float] = (2.0, 1.0)
    start_position: tuple[float, float] | None = None
    foreground_value: float = 1.0
    background_value: float = 0.0


def generate_moving_circle_frames(config: MovingCircleConfig) -> ArrayTHW:
    """Generate a [T, H, W] sequence of a circle bouncing around inside a grid.

    The circle reflects off the grid edges rather than wrapping around them, so it
    always stays fully inside the frame and is never split across edges/corners.
    """
    height, width = config.height, config.width
    row, col = config.start_position or (height / 2, width / 2)
    v_row, v_col = config.velocity
    row_margin, col_margin = config.radius, config.radius
    row_max, col_max = height - 1 - row_margin, width - 1 - col_margin

    row_grid, col_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    frames = np.empty((config.n_timesteps, height, width), dtype=np.float32)
    for t in range(config.n_timesteps):
        inside = (row_grid - row) ** 2 + (col_grid - col) ** 2 <= config.radius**2
        frames[t] = np.where(inside, config.foreground_value, config.background_value)

        row, col = row + v_row, col + v_col
        if row < row_margin or row > row_max:
            v_row = -v_row
            row = np.clip(row, row_margin, row_max)
        if col < col_margin or col > col_max:
            v_col = -v_col
            col = np.clip(col, col_margin, col_max)

    return frames
