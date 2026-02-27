"""SIEError metric.

Adapted from the IceNet repository at:
- https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb.
"""

import torch
from torchmetrics import Metric

SEA_ICE_THRESHOLD = 0.15  # Threshold for defining sea ice extent


class SIEError(Metric):
    """Sea Ice Extent error metric (in km^2) for use at multiple lead times."""

    def __init__(self, leadtimes_to_evaluate: list[int], pixel_size: int = 25) -> None:
        """Initialize the SIEError metric.

        Parameters
        ----------
        leadtimes_to_evaluate: List[int]
            Indices of lead times at which SIE should be evaluated.
        pixel_size: int, optional
            Physical size of one pixel in kilometers (default is 25 km -> OSISAF).

        """
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state("pred_sie", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("true_sie", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.pixel_size = pixel_size

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        _sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Update the SIE accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth values.
        sample_weight : Optional[torch.Tensor]
            Ignored (present for API compatibility).

        """
        preds = (preds > SEA_ICE_THRESHOLD).long()
        target = (target > SEA_ICE_THRESHOLD).long()

        self.pred_sie += torch.sum(preds[:, :, :, self.leadtimes_to_evaluate])  # type: ignore[operator]
        self.true_sie += torch.sum(target[:, :, :, self.leadtimes_to_evaluate])  # type: ignore[operator]

    def compute(self) -> torch.Tensor:
        """Compute the final Sea Ice Extent error in km²."""
        return (self.pred_sie - self.true_sie) * self.pixel_size**2  # type: ignore[operator]
