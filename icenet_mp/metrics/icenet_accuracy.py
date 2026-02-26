"""IceNetAccuracy metric.

Adapted from the IceNet repository at:
- https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb
"""

import torch
from torchmetrics import Metric

THRESHOLD = 0.15  # Threshold for binarizing predictions and targets


class IceNetAccuracy(Metric):
    """Binary accuracy metric for use at multiple leadtimes."""

    def __init__(self, leadtimes_to_evaluate: list) -> None:
        """Initialize the IceNetAccuracy metric."""
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state(
            "weighted_score", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "possible_score", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor
    ) -> None:
        """Update metric state with a new batch of predictions and targets."""
        preds = (preds > THRESHOLD).long()
        target = (target > THRESHOLD).long()
        sample_weight = sample_weight.squeeze(-1)
        base_score = (
            preds[:, :, :, self.leadtimes_to_evaluate]
            == target[:, :, :, self.leadtimes_to_evaluate]
        )
        self.weighted_score += torch.sum(  # type: ignore[operator]
            base_score * sample_weight[:, :, :, self.leadtimes_to_evaluate]
        )
        self.possible_score += torch.sum(  # type: ignore[operator]
            sample_weight[:, :, :, self.leadtimes_to_evaluate]
        )

    def compute(self) -> torch.Tensor:
        """Compute the final accuracy metric as a percentage."""
        return self.weighted_score.float() / self.possible_score * 100.0  # type: ignore[arg-type, operator]
