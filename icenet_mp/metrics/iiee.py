"""IntegratedIceEdgeError (IIEE) metric."""

import torch
from torchmetrics import Metric

SEA_ICE_THRESHOLD = 0.15  # Threshold for defining ice/no-ice, and hence the ice edge


class IntegratedIceEdgeErrorPerForecastDay(Metric):
    """Integrated Ice Edge Error (IIEE) metric (in km^2) for use at multiple lead times.

    IIEE is the area of the symmetric difference between the predicted and true ice
    extent: the total area where the two disagree on ice presence, following
    Goessling et al. (2016, https://doi.org/10.1002/2015GL067232). Sea ice presence
    is defined by having a probability greater than the threshold value.

    Unlike `SeaIceExtentErrorPerForecastDay`, which is a signed difference of extents
    and can cancel out over- and under-estimation, IIEE always accumulates
    disagreement and is therefore always >= |SIEError|.
    """

    def __init__(
        self, pixel_size: int = 25, land_mask: torch.Tensor | None = None
    ) -> None:
        """Initialize the IIEE metric.

        Parameters
        ----------
        pixel_size: int, optional
            Physical size of one pixel in kilometers (default is 25 km -> OSISAF).
        land_mask: torch.Tensor, optional
            Boolean tensor of shape (H, W), True for ocean cells and False for land.
            When given, land cells never count toward disagreement.

        """
        super().__init__()
        self.sum_errors: torch.Tensor
        self.sample_count: torch.Tensor
        self.pixel_size = pixel_size
        if land_mask is not None:
            self.register_buffer("land_mask", land_mask.bool(), persistent=False)

        # States initialized lazily on first update
        self.add_state("sum_errors", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("sample_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Update the IIEE accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions of shape (B, T, C, H, W).
        target : torch.Tensor
            Ground truth values of shape (B, T, C, H, W).

        """
        preds_extent = preds > SEA_ICE_THRESHOLD
        target_extent = target > SEA_ICE_THRESHOLD

        # Per-sample count of pixels where extent disagrees (B, T)
        disagreement = (preds_extent != target_extent).float()
        land_mask = getattr(self, "land_mask", None)
        if land_mask is not None:
            disagreement = disagreement * land_mask.to(dtype=disagreement.dtype)
        error = torch.sum(disagreement, dim=(2, 3, 4))

        # Initialize states on first update
        if self.sum_errors.numel() == 0:
            self.sum_errors = error.sum(dim=0)
        else:
            # Accumulate sums and counts per lead time
            self.sum_errors += error.sum(dim=0)  # Sum across batch dimension
        self.sample_count += error.shape[0]  # Increment count by batch size

    def compute(self) -> torch.Tensor:
        """Compute the final Integrated Ice Edge Error in km²."""
        if self.sum_errors.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        mean_error = self.sum_errors / self.sample_count
        return mean_error * self.pixel_size**2  # type: ignore[operator]
