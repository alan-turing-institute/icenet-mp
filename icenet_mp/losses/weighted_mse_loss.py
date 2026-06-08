"""Weighted MSELoss.

Adapted from the IceNet repository at:
- https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb
"""

from typing import Any

from torch import Tensor, nn


class WeightedMSELoss(nn.MSELoss):
    """Mean-squared error loss with per-element weighting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the WeightedMSELoss.

        Args:
            *args: Positional arguments passed to torch.nn.MSELoss.
            **kwargs: Keyword arguments passed to torch.nn.MSELoss.

        """
        kwargs["reduction"] = "none"
        super().__init__(*args, **kwargs)

    def forward(  # type: ignore[override]
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        """Compute weighted mean squared error loss.

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
