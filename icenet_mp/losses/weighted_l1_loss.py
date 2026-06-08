"""Weighted L1Loss.

Adapted from the IceNet repository: https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb
"""

from typing import Any

from torch import Tensor, nn


class WeightedL1Loss(nn.L1Loss):
    """L1 loss with elementwise weighting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the WeightedL1Loss.

        Args:
            *args: Positional arguments passed to torch.nn.L1Loss.
            **kwargs: Keyword arguments passed to torch.nn.L1Loss.

        """
        super().__init__(*args, reduction="none", **kwargs)

    def forward(  # type: ignore[override]
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None = None,
    ) -> Tensor:  # type: ignore[override]
        """Compute weighted L1 loss.

        Args:
            preds (Tensor): Predicted values.
            targets (Tensor): Ground-truth values.
            sample_weights (Tensor | None): Elementwise weighting tensor. If None, no weighting is applied.

        Returns:
            Tensor: Scalar weighted loss value.

        """
        y_hat = preds.squeeze()
        targets = targets.squeeze()
        loss = super().forward(y_hat, targets)
        if sample_weights is not None:
            loss = loss * sample_weights.squeeze()
        return loss.mean()
