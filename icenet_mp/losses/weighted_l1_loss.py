"""Weighted L1Loss.

Adapted from the IceNet repository: https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb
"""

from typing import Any

import torch
from torch import Tensor, nn


class WeightedL1Loss(nn.L1Loss):
    """L1 loss with elementwise weighting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the WeightedL1Loss.

        Args:
            *args: Positional arguments passed to torch.nn.L1Loss.
            **kwargs: Keyword arguments passed to torch.nn.L1Loss.

        """
        super().__init__(*args, **kwargs)

    def forward(  # type: ignore[override]
        self,
        inputs: Tensor,
        targets: Tensor,
        sample_weights: Tensor,
    ) -> Tensor:  # type: ignore[override]
        """Compute weighted L1 loss.

        The loss is computed as elementwise L1 error multiplied by
        `sample_weights`, then averaged to a scalar.
        """
        y_hat = torch.sigmoid(inputs)
        loss = super().forward(100 * y_hat, 100 * targets) * sample_weights
        return loss.mean()
