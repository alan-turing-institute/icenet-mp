"""SIEError metric."""

import torch
from torchmetrics import Metric

SEA_ICE_THRESHOLD = 0.15  # Threshold for defining sea ice extent


class SeaIceExtentErrorPerForecastDay(Metric):
    """Sea Ice Extent error (SIEError) metric (in km^2) for use at multiple lead times.

    The SIE error is calculated as the difference between the predicted and
    true sea ice extent for each forecast day. Sea ice presence is defined by having
    a concentration greater than the threshold value.
    """

    def __init__(
        self, pixel_size: int = 25, land_mask: torch.Tensor | None = None
    ) -> None:
        """Initialize the SIEError metric.

        Parameters
        ----------
        pixel_size: int, optional
            Physical size of one pixel in kilometers (default is 25 km -> OSISAF).
        land_mask: torch.Tensor, optional
            Boolean tensor of shape (H, W), True for ocean cells and False for land.
            When given, land cells never count toward extent on either side.

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
        """Update the SIE accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions of shape (B, T, C, H, W).
        target : torch.Tensor
            Ground truth values of shape (B, T, C, H, W).

        """
        preds = preds > SEA_ICE_THRESHOLD
        target = target > SEA_ICE_THRESHOLD
        land_mask = getattr(self, "land_mask", None)
        if land_mask is not None:
            preds = preds & land_mask
            target = target & land_mask

        # Calculate the SIE for each day of the forecast
        pred_sie = torch.sum(preds, dim=(2, 3, 4))  # Shape: (B, T)
        true_sie = torch.sum(target, dim=(2, 3, 4))  # Shape: (B, T)

        # Per-sample SIE error (B, T)
        error = (pred_sie - true_sie).float()

        # Initialize states on first update
        if self.sum_errors.numel() == 0:
            self.sum_errors = error.sum(
                dim=0
            )  # torch.zeros(error.shape[1], device=self.device)
        else:
            # Accumulate sums and counts per lead time
            self.sum_errors += error.sum(dim=0)  # Sum across batch dimension
        self.sample_count += error.shape[0]  # Increment count by batch size

    def compute(self) -> torch.Tensor:
        """Compute the final Sea Ice Extent error in km²."""
        if self.sum_errors.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        mean_error = self.sum_errors / self.sample_count
        return mean_error * self.pixel_size**2  # type: ignore[operator]
