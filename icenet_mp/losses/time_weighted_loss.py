"""Lead-time weighted loss wrapper."""

from typing import ClassVar

import torch
from torch import nn


class TimeWeightedLoss(nn.Module):
    """Apply increasing linear weights across forecast lead times."""

    requires_time_dimension: ClassVar[bool] = True
    minimum_input_dims: ClassVar[int] = 3

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        initial_weight: float = 1.0,
        final_weight: float = 2.0,
    ) -> None:
        """Initialise a time-weighted wrapper around a scalar base loss."""
        super().__init__()
        if not isinstance(base_loss, nn.Module):
            msg = (
                f"base_loss must be a torch.nn.Module, got {type(base_loss).__name__}."
            )
            raise TypeError(msg)
        if initial_weight <= 0:
            msg = "initial_weight must be greater than 0."
            raise ValueError(msg)
        if final_weight < initial_weight:
            msg = "final_weight must be greater than or equal to initial_weight."
            raise ValueError(msg)

        self.base_loss = base_loss
        self.initial_weight = float(initial_weight)
        self.final_weight = float(final_weight)

    def _scalar_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Return a scalar from the wrapped loss."""
        loss = self.base_loss(prediction, target)
        return loss.mean() if loss.ndim else loss

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combine per-step losses with increasing lead-time weights."""
        if prediction.shape != target.shape:
            msg = (
                f"prediction shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(target.shape)}."
            )
            raise ValueError(msg)
        if prediction.ndim < self.minimum_input_dims:
            msg = (
                "TimeWeightedLoss expects shape [B, T, ...], got "
                f"{tuple(prediction.shape)}."
            )
            raise ValueError(msg)

        n_forecast_steps = prediction.shape[1]
        if n_forecast_steps == 0:
            msg = "TimeWeightedLoss cannot use an empty forecast-time dimension."
            raise ValueError(msg)

        if n_forecast_steps == 1 or self.initial_weight == self.final_weight:
            return self._scalar_loss(prediction, target)

        step_losses = torch.stack(
            [
                self._scalar_loss(prediction[:, step], target[:, step])
                for step in range(n_forecast_steps)
            ]
        )
        weights = torch.linspace(
            self.initial_weight,
            self.final_weight,
            n_forecast_steps,
            device=step_losses.device,
            dtype=step_losses.dtype,
        )
        weights = weights / weights.mean()
        return (step_losses * weights).mean()
