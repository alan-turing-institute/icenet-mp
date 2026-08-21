"""IceNetAccuracy metric.

Adapted from the IceNet implementation at:
- https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb
"""

import torch
from torchmetrics import Metric

SEA_ICE_THRESHOLD = 0.15  # Threshold for binarizing predictions and targets


class IceNetAccuracyPerForecastDay(Metric):
    """Binary accuracy metric for use at multiple leadtimes."""

    def __init__(self, land_mask: torch.Tensor | None = None) -> None:
        """Initialize the IceNetAccuracy metric.

        Parameters
        ----------
        land_mask : torch.Tensor, optional
            Boolean tensor of shape (H, W), True for ocean cells and False for land.
            When given, land cells are excluded from the accuracy calculation
            entirely, rather than counted as trivially-correct "no ice" agreements.

        """
        super().__init__()
        if land_mask is not None:
            self.register_buffer("land_mask", land_mask.bool(), persistent=False)
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
        preds = (preds > SEA_ICE_THRESHOLD).long()
        target = (target > SEA_ICE_THRESHOLD).long()
        if sample_weight is None:
            sample_weight = torch.ones_like(target)
        land_mask = getattr(self, "land_mask", None)
        if land_mask is not None:
            sample_weight = sample_weight * land_mask.to(dtype=sample_weight.dtype)
        base_score = preds == target
        weighted_score = torch.sum(base_score * sample_weight, dim=[0, 2, 3, 4])
        if self.weighted_score.numel() == 0:  # type: ignore[has-type]
            self.weighted_score = weighted_score
        else:
            self.weighted_score += weighted_score
        possible_score = torch.sum(sample_weight, dim=[0, 2, 3, 4])
        if self.possible_score.numel() == 0:  # type: ignore[has-type]
            self.possible_score = possible_score
        else:
            self.possible_score += possible_score

    def compute(self) -> torch.Tensor:
        """Compute the final accuracy metric as a percentage at each leadtime."""
        return self.weighted_score.float() / self.possible_score * 100.0  # type: ignore[arg-type, operator]
