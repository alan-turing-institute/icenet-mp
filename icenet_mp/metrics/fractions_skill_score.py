"""Fractions Skill Score for thresholded sea-ice concentration forecasts."""

import torch
from torchmetrics import Metric


class FractionsSkillScorePerForecastDay(Metric):
    """Compute Fractions Skill Score independently for each forecast lead time.

    Sea-ice concentration is thresholded to a binary ice/no-ice field, converted to
    neighbourhood fractions with average pooling, and scored using the standard FSS
    definition. A score of 1 is perfect and 0 indicates no neighbourhood skill.
    """

    numerator: torch.Tensor
    denominator: torch.Tensor

    def __init__(
        self,
        *,
        threshold: float = 0.15,
        window_size: int = 3,
    ) -> None:
        """Initialise the metric.

        Args:
            threshold: SIC threshold used to define sea ice.
            window_size: Odd neighbourhood width in grid cells.
        """
        super().__init__()
        if not 0.0 <= threshold <= 1.0:
            msg = "threshold must be between 0 and 1."
            raise ValueError(msg)
        if window_size <= 0 or window_size % 2 == 0:
            msg = "window_size must be a positive odd integer."
            raise ValueError(msg)

        self.threshold = threshold
        self.window_size = window_size
        self.add_state(
            "numerator",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "denominator",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore[override]
        """Accumulate FSS sufficient statistics for a batch of NTCHW tensors."""
        if preds.shape != target.shape:
            msg = f"Expected matching prediction/target shapes, got {preds.shape} and {target.shape}."
            raise ValueError(msg)
        if preds.ndim != 5:  # noqa: PLR2004
            msg = f"Expected NTCHW tensors with 5 dimensions, got {preds.ndim}."
            raise ValueError(msg)

        batch_size, n_steps, channels, height, width = preds.shape
        reshape = (batch_size * n_steps, channels, height, width)
        padding = self.window_size // 2

        pred_binary = (preds >= self.threshold).float().reshape(reshape)
        target_binary = (target >= self.threshold).float().reshape(reshape)
        pred_fraction = torch.nn.functional.avg_pool2d(
            pred_binary,
            kernel_size=self.window_size,
            stride=1,
            padding=padding,
        ).reshape(batch_size, n_steps, channels, height, width)
        target_fraction = torch.nn.functional.avg_pool2d(
            target_binary,
            kernel_size=self.window_size,
            stride=1,
            padding=padding,
        ).reshape(batch_size, n_steps, channels, height, width)

        reduce_dims = (0, 2, 3, 4)
        numerator = ((pred_fraction - target_fraction) ** 2).sum(dim=reduce_dims)
        denominator = (pred_fraction.square() + target_fraction.square()).sum(
            dim=reduce_dims
        )

        if self.numerator.numel() == 0:
            self.numerator = numerator
            self.denominator = denominator
        elif self.numerator.shape != numerator.shape:
            msg = (
                f"Time dimension mismatch: expected {self.numerator.shape[0]}, "
                f"got {numerator.shape[0]}."
            )
            raise ValueError(msg)
        else:
            self.numerator += numerator
            self.denominator += denominator

    def compute(self) -> torch.Tensor:
        """Return FSS for each forecast lead time."""
        if self.numerator.numel() == 0:
            return torch.tensor([], dtype=torch.float32, device=self.numerator.device)
        return torch.where(
            self.denominator > 0,
            1.0 - self.numerator / self.denominator,
            torch.ones_like(self.denominator),
        )
