"""Distance-averaged Integrated Ice Edge Error (DIIEE) metric.

Adapted from the ``diiee_avg_displacement`` calculation in a MET Norway ice-chart
verification pipeline (``diiee_avg_calculation.py`` / ``IIEERecord.py``), which
compares an automatic ice product against a manually-drawn ice chart polygon using
vector geometry. Here the comparison is against the gridded ground-truth satellite
SIC field already used by every other metric in this package, and ice edges are
detected on the raster grid (`icenet_mp.metrics.ice_edge.binary_edge`, shared with
`icenet_mp.metrics.fss`) rather than via vector polygon boundaries.

The original pipeline also excludes, from each edge-length calculation, any polygon
boundary that coincides with the edge of the satellite product's coverage extent (so
a "fake" ice edge created by clipping to the swath isn't counted as real). IceNet-MP
grids have no such clipped-extent concept -- the ice edge measured here therefore
does not exclude the grid's own domain boundary, consistent with the (pre-existing)
`icenet_mp.metrics.fss` edge detection, which has the same simplification.
"""

import torch
from torchmetrics import Metric

from .ice_edge import binary_edge

SEA_ICE_THRESHOLD = 0.15  # Threshold for defining ice/no-ice, and hence the ice edge


class DistanceAveragedIceEdgeErrorPerForecastDay(Metric):
    """Distance-averaged Integrated Ice Edge Error (DIIEE), in km, per lead time.

    The total misclassified-ice area (overestimation + underestimation, i.e. IIEE,
    see `icenet_mp.metrics.iiee`) is normalised by the combined length of the
    predicted and true ice edges, giving an average displacement distance in km:
    roughly, how far the ice edge would need to move to reconcile the two fields.

        DIIEE = 2 * (over_area + under_area) / (pred_edge_length + true_edge_length)

    Edge length is approximated on the raster grid as
    ``(edge cell count) * pixel_size``; this is coarser than a true vector polygon
    perimeter but requires no vector geometry or land-mask polygon. Lead times where
    both fields are entirely ice or entirely ice-free (combined edge length zero) are
    undefined and reported as NaN, rather than the ``-9999.99`` sentinel used
    upstream, to compose correctly with tensor reductions (e.g. `nanmean`).
    """

    def __init__(self, pixel_size: int = 25) -> None:
        """Initialize the DIIEE metric.

        Parameters
        ----------
        pixel_size: int, optional
            Physical size of one pixel in kilometers (default is 25 km -> OSISAF).

        """
        super().__init__()
        self.pixel_size = pixel_size

        self.sum_mismatch_area: torch.Tensor
        self.sum_edge_length: torch.Tensor

        # States initialized lazily on first update
        self.add_state(
            "sum_mismatch_area",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_edge_length",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Update the DIIEE accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions of shape (B, T, C, H, W).
        target : torch.Tensor
            Ground-truth satellite SIC of shape (B, T, C, H, W).

        """
        batch_size, n_steps, n_channels, height, width = preds.shape

        preds_extent = (preds > SEA_ICE_THRESHOLD).reshape(-1, height, width)
        target_extent = (target > SEA_ICE_THRESHOLD).reshape(-1, height, width)

        # Combined over- and under-estimation area (the IIEE integral)
        mismatch = (preds_extent != target_extent).sum(dim=(-2, -1)).float()
        mismatch_area = (
            mismatch.view(batch_size, n_steps, n_channels).sum(dim=(0, 2))
            * self.pixel_size**2
        )

        # Combined predicted + true ice-edge length
        pred_edge_cells = binary_edge(preds_extent).sum(dim=(-2, -1)).float()
        target_edge_cells = binary_edge(target_extent).sum(dim=(-2, -1)).float()
        edge_length = (pred_edge_cells + target_edge_cells).view(
            batch_size, n_steps, n_channels
        ).sum(dim=(0, 2)) * self.pixel_size

        if self.sum_mismatch_area.numel() == 0:
            self.sum_mismatch_area = mismatch_area
            self.sum_edge_length = edge_length
        else:
            self.sum_mismatch_area += mismatch_area
            self.sum_edge_length += edge_length

    def compute(self) -> torch.Tensor:
        """Compute the final DIIEE (average ice-edge displacement, in km) per lead time."""
        if self.sum_mismatch_area.numel() == 0:
            return torch.tensor(
                [], dtype=torch.float32, device=self.sum_mismatch_area.device
            )

        safe_edge_length = self.sum_edge_length.clamp(min=1e-8)
        diiee = 2.0 * self.sum_mismatch_area / safe_edge_length
        return torch.where(
            self.sum_edge_length > 0, diiee, torch.full_like(diiee, float("nan"))
        )
