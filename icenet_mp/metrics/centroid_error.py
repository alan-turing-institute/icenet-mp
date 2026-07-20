"""CentroidError metric: pixel distance between predicted and target centroids."""

import torch
from torchmetrics import Metric

# Frames whose target has less total mass than this are treated as empty (undefined
# centroid) and excluded from the average; it also floors the denominator so the
# centroid of an empty frame never divides by zero.
_EMPTY_MASS_THRESHOLD = 1e-8


class CentroidErrorPerForecastDay(Metric):
    """Euclidean distance (in pixels) between the predicted and target centroids.

    The centroid of a (batch, time) frame is its value-weighted center of mass over
    the spatial dimensions, summed across channels. Frames whose target has
    (near-)zero total mass have an undefined centroid and are excluded from the
    average.
    """

    def __init__(self) -> None:
        """Initialize the metric."""
        super().__init__()
        self.sum_errors: torch.Tensor
        self.count: torch.Tensor
        self.add_state(
            "sum_errors",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.tensor([], dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @staticmethod
    def _centroids(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (row, col) centroids and total mass for each (batch, time) frame.

        Args:
            values: Non-negative tensor of shape (batch, time, channels, height, width).

        Returns:
            centroids: Tensor of shape (batch, time, 2) with (row, col) centroid.
            mass: Tensor of shape (batch, time) with total mass, for masking empty frames.

        """
        weight = values.sum(dim=2)  # (batch, time, height, width)
        height, width = weight.shape[-2], weight.shape[-1]
        row_idx = torch.arange(height, device=weight.device, dtype=weight.dtype)
        col_idx = torch.arange(width, device=weight.device, dtype=weight.dtype)
        mass = weight.sum(dim=(-2, -1))  # (batch, time)
        safe_mass = mass.clamp(min=_EMPTY_MASS_THRESHOLD)
        row_centroid = (weight.sum(dim=-1) * row_idx).sum(dim=-1) / safe_mass
        col_centroid = (weight.sum(dim=-2) * col_idx).sum(dim=-1) / safe_mass
        return torch.stack([row_centroid, col_centroid], dim=-1), mass

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state with a new batch of predictions and targets."""
        pred_centroids, _ = self._centroids(preds.clamp(min=0))
        target_centroids, target_mass = self._centroids(target.clamp(min=0))

        distances = torch.linalg.norm(pred_centroids - target_centroids, dim=-1)
        valid = (target_mass > _EMPTY_MASS_THRESHOLD).float()

        batch_sum_errors = (distances * valid).sum(dim=0)
        batch_count = valid.sum(dim=0).long()

        if self.sum_errors.numel() == 0:
            self.sum_errors = batch_sum_errors
            self.count = batch_count
        else:
            self.sum_errors += batch_sum_errors
            self.count += batch_count

    def compute(self) -> torch.Tensor:
        """Compute the mean centroid distance (in pixels) per forecast lead time."""
        if self.count.numel() == 0:
            return torch.tensor([], dtype=torch.float32, device=self.sum_errors.device)
        count = torch.clamp(self.count, min=1)
        return self.sum_errors / count.float()
