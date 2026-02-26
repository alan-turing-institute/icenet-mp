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
        super().__init__(*args, **kwargs)

    def forward(  # type: ignore[override]
        self,
        inputs: Tensor,
        targets: Tensor,
        sample_weights: Tensor,
    ) -> Tensor:  # type: ignore[override]
        """Weighted BCEWithLogitsLoss.

        Compute BCEWithLogitsLoss weighted by masking.
        Using BCEWithLogitsLoss instead of BCELoss, as it is more numerically stable:
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        """
        loss = super().forward(
            inputs.movedim(-2, 1),
            targets.movedim(-1, 1),
        ) * sample_weights.movedim(-1, 1)

        return loss.mean()
