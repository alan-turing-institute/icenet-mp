"""Weighted BCEWithLogitsLoss.

Adapted from the IceNet repository at:
- https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb
"""

from typing import Any

from torch import Tensor, nn


class WeightedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """BCEWithLogits loss with elementwise weighting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the WeightedBCEWithLogitsLoss.

        Args:
            *args: Positional arguments passed to torch.nn.BCEWithLogitsLoss.
            **kwargs: Keyword arguments passed to torch.nn.BCEWithLogitsLoss.

        """
        if "reduction" in kwargs and kwargs["reduction"] != "none":
          log.warning(
              "Ignoring reduction='%s'; this loss requires reduction='none' "
              "for weighted loss computation.", kwargs["reduction"]
           )
        kwargs["reduction"] = "none"
        super().__init__(*args, **kwargs)

    def forward(  # type: ignore[override]
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None = None,
    ) -> Tensor:  # type: ignore[override]
        """Compute weighted BCEWithLogits loss.

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
