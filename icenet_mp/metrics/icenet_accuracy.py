"""IceNetAccuracy metric.

Adapted from the IceNet implementation at:
- https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb
"""

import torch
from torchmetrics import Metric

THRESHOLD = 0.15  # Threshold for binarizing predictions and targets


class IceNetAccuracy(Metric):
    """Binary accuracy metric for use at multiple leadtimes."""

    def __init__(self) -> None:
        """Initialize the IceNetAccuracy metric."""
        super().__init__()
        self.add_state(
            "weighted_score",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "possible_score",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Update metric state with a new batch of predictions and targets."""
        preds = (preds > THRESHOLD).long()
        target = (target > THRESHOLD).long()
        base_score = preds == target
        if sample_weight is None:
            # No weights: the weighted sum is just the match count and the possible
            # score is the element count, without materialising a tensor of ones.
            weighted_score = torch.sum(base_score, dim=[0, 2, 3, 4])
            possible_score = torch.full_like(
                weighted_score,
                base_score.numel() // base_score.shape[1],
            )
        else:
            weighted_score = torch.sum(base_score * sample_weight, dim=[0, 2, 3, 4])
            possible_score = torch.sum(sample_weight, dim=[0, 2, 3, 4])
        if self.weighted_score.numel() == 0:  # type: ignore[has-type]
            self.weighted_score = weighted_score
        else:
            self.weighted_score += weighted_score
        if self.possible_score.numel() == 0:  # type: ignore[has-type]
            self.possible_score = possible_score
        else:
            self.possible_score += possible_score

    def compute(self) -> torch.Tensor:
        """Compute the final accuracy metric as a percentage at each leadtime."""
        return self.weighted_score.float() / self.possible_score * 100.0  # type: ignore[arg-type, operator]
