"""Calculating RMSE, MAE by forecast step."""

import torch
from torchmetrics import Metric


class BaseErrorMetricDaily(Metric):
    """Base class for per-timestep error metrics using sufficient statistics."""

    def __init__(self) -> None:
        """Initialize the metric."""
        super().__init__()
        self.sum_errors: torch.Tensor
        self.count: torch.Tensor
        self.add_state(
            "sum_errors",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.tensor([], dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def _compute_errors(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute element-wise errors. Override in subclasses."""
        raise NotImplementedError

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with a batch of predictions and targets.

        Args:
            preds: Tensor of shape (batch, time, channels, height, width)
            targets: Tensor of shape (batch, time, channels, height, width)

        """
        # Compute errors: (batch, time, channels, height, width)
        errors = self._compute_errors(preds, targets)

        batch_size = errors.shape[0]
        num_spatial = errors.shape[2] * errors.shape[3] * errors.shape[4]

        # Reshape to (batch, time, -1) then sum over batch and spatial dims
        errors_reshaped = errors.view(batch_size, -1, num_spatial)
        batch_sum_errors = errors_reshaped.sum(dim=(0, 2))

        # Count samples per time step
        batch_count = torch.full(
            (errors.shape[1],),
            batch_size * num_spatial,
            dtype=torch.long,
            device=errors.device,
        )

        # Initialize buffers on first update
        if self.sum_errors.numel() == 0:
            self.sum_errors = batch_sum_errors
            self.count = batch_count
        else:
            if self.sum_errors.shape[0] != batch_sum_errors.shape[0]:
                msg = f"Time dimension mismatch: expected {self.sum_errors.shape[0]}, got {batch_sum_errors.shape[0]}"
                raise ValueError(msg)
            self.sum_errors += batch_sum_errors
            self.count += batch_count

    def _finalize(self, mean_errors: torch.Tensor) -> torch.Tensor:
        """Apply final transformation to mean errors. Override in subclasses."""
        return mean_errors

    def compute(self) -> torch.Tensor:
        """Compute metric per lead time from accumulated sufficient statistics.

        Returns:
            Tensor of shape (T,) with metric value for each time step

        """
        if self.count.numel() == 0:
            return torch.tensor([], dtype=torch.float32, device=self.sum_errors.device)

        count = torch.clamp(self.count, min=1)
        mean_errors = self.sum_errors / count.float()
        return self._finalize(mean_errors)


class RMSEDaily(BaseErrorMetricDaily):
    """Root Mean Squared Error per forecast lead time."""

    def _compute_errors(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (preds - targets) ** 2

    def _finalize(self, mean_errors: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(mean_errors)


class MAEDaily(BaseErrorMetricDaily):
    """Mean Absolute Error per forecast lead time."""

    def _compute_errors(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.abs(preds - targets)
