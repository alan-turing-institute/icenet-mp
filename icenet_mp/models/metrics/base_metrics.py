"""Calculating RMSE, MAE by forecast step."""

import torch
from torchmetrics import Metric


class RMSEDaily(Metric):
    def __init__(self) -> None:
        """Initialize the RMSEDaily metric."""
        super().__init__()
        # Register buffers for sufficient statistics per time step
        # Shape: (T,) where T is the number of lead times
        self.sum_squared_errors: torch.Tensor
        self.count: torch.Tensor
        self.add_state(
            "sum_squared_errors",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.tensor([], dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with a batch of predictions and targets.

        Args:
            preds: Tensor of shape (batch, time, channels, height, width)
            targets: Tensor of shape (batch, time, channels, height, width)

        """
        # Compute squared errors: (batch, time, channels, height, width)
        squared_errors = (preds - targets) ** 2

        # Sum over all dimensions except time to get (batch, time)
        batch_size = squared_errors.shape[0]
        num_spatial = (
            squared_errors.shape[2] * squared_errors.shape[3] * squared_errors.shape[4]
        )

        # Reshape to (batch, time, -1) then sum over batch and spatial dims
        squared_errors_reshaped = squared_errors.view(batch_size, -1, num_spatial)
        # Sum per time step: (time,)
        batch_sum_squared_errors = squared_errors_reshaped.sum(dim=(0, 2))

        # Count samples per time step: batch_size * num_spatial samples per time step
        batch_count = torch.full(
            (squared_errors.shape[1],),
            batch_size * num_spatial,
            dtype=torch.long,
            device=squared_errors.device,
        )

        # Initialize buffers on first update
        if self.sum_squared_errors.numel() == 0:
            self.sum_squared_errors = batch_sum_squared_errors
            self.count = batch_count
        else:
            # Ensure shapes match (in case time dimension varies)
            if self.sum_squared_errors.shape[0] != batch_sum_squared_errors.shape[0]:
                msg = f"Time dimension mismatch: expected {self.sum_squared_errors.shape[0]}, got {batch_sum_squared_errors.shape[0]}"
                raise ValueError(msg)
            # Accumulate sufficient statistics
            self.sum_squared_errors += batch_sum_squared_errors
            self.count += batch_count

    def compute(self) -> torch.Tensor:
        """Compute RMSE per lead time from accumulated sufficient statistics.

        Returns:
            Tensor of shape (T,) with RMSE for each time step

        """
        if self.count.numel() == 0:
            return torch.tensor(
                [], dtype=torch.float32, device=self.sum_squared_errors.device
            )

        # Avoid division by zero
        count = torch.clamp(self.count, min=1)

        # Calculate RMSE: sqrt(sum_squared_errors / count)
        return torch.sqrt(self.sum_squared_errors / count.float())


class MAEDaily(Metric):
    def __init__(self) -> None:
        """Initialize the MAEDaily metric."""
        super().__init__()
        self.mae_daily: torch.Tensor
        self.add_state("mae_daily", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        _sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Update the MAEDaily accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth values.
        sample_weight : Optional[torch.Tensor]
            Ignored (present for API compatibility).

        """
        mae = torch.mean(torch.abs(preds - target), dim=(0, 2, 3, 4))
        if self.mae_daily.numel() == 0:
            self.mae_daily = mae.unsqueeze(1)  # Shape: (T,)
        else:
            self.mae_daily = torch.cat(
                (self.mae_daily, mae.unsqueeze(1)), dim=1
            )  # Shape: (T, N)

    def compute(self) -> torch.Tensor:
        """Compute the final MAEDaily values."""
        return torch.mean(self.mae_daily, dim=1)
