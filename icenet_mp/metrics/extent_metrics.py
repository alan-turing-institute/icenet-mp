"""Threshold-based sea ice extent/edge metrics: SIE error, IIEE, and DIIEE.

Adapted in part from the ``diiee_avg_displacement`` calculation in a MET Norway
ice-chart verification pipeline (``diiee_avg_calculation.py`` / ``IIEERecord.py``),
which compares an automatic ice product against a manually-drawn ice chart polygon
using vector geometry. Here the comparison is against the gridded ground-truth
satellite SIC field already used by every other metric in this package, and ice
edges are detected on the raster grid (`icenet_mp.metrics.ice_edge.binary_edge`,
shared with `icenet_mp.metrics.fss`) rather than via vector polygon boundaries.

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


class _IceAreaMetricBase(Metric):
    """Shared construction for threshold-based sea ice extent/edge metrics."""

    def __init__(
        self, pixel_size: int = 25, land_mask: torch.Tensor | None = None
    ) -> None:
        """Initialize shared state.

        Parameters
        ----------
        pixel_size: int, optional
            Physical size of one pixel in kilometers (default is 25 km -> OSISAF).
        land_mask: torch.Tensor, optional
            Boolean tensor of shape (H, W), True for ocean cells and False for land.
            When given, land cells are excluded from the metric entirely.

        """
        super().__init__()
        self.pixel_size = pixel_size
        if land_mask is not None:
            self.register_buffer("land_mask", land_mask.bool(), persistent=False)

    def _masked_mismatch(
        self, preds_extent: torch.Tensor, target_extent: torch.Tensor
    ) -> torch.Tensor:
        """Land-masked disagreement map, same shape as the inputs (..., H, W)."""
        mismatch = (preds_extent != target_extent).float()
        land_mask = getattr(self, "land_mask", None)
        if land_mask is not None:
            mismatch = mismatch * land_mask.to(dtype=mismatch.dtype)
        return mismatch


class _MeanIceAreaMetric(_IceAreaMetricBase):
    """Shared accumulation for metrics reported as ``pixel_size**2 * mean(error)``.

    Subclasses supply the per-(batch, lead-time) error contribution via
    ``_batch_error()``; this class handles the running sum/count and final scaling.
    """

    def __init__(
        self, pixel_size: int = 25, land_mask: torch.Tensor | None = None
    ) -> None:
        """Initialize the metric state (see `_IceAreaMetricBase` for parameters)."""
        super().__init__(pixel_size=pixel_size, land_mask=land_mask)
        self.sum_errors: torch.Tensor
        self.sample_count: torch.Tensor

        # States initialized lazily on first update
        self.add_state("sum_errors", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("sample_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _batch_error(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-(batch, lead-time) error contribution, shape (B, T). Override in subclasses."""
        raise NotImplementedError

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the accumulators with a new batch.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions of shape (B, T, C, H, W).
        target : torch.Tensor
            Ground truth values of shape (B, T, C, H, W).

        """
        error = self._batch_error(preds, target)

        # Initialize states on first update
        if self.sum_errors.numel() == 0:
            self.sum_errors = error.sum(dim=0)
        else:
            # Accumulate sums and counts per lead time
            self.sum_errors += error.sum(dim=0)  # Sum across batch dimension
        self.sample_count += error.shape[0]  # Increment count by batch size

    def compute(self) -> torch.Tensor:
        """Compute the final metric in km²."""
        if self.sum_errors.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        mean_error = self.sum_errors / self.sample_count
        return mean_error * self.pixel_size**2  # type: ignore[operator]


class SeaIceExtentErrorPerForecastDay(_MeanIceAreaMetric):
    """Sea Ice Extent error (SIEError) metric (in km^2) for use at multiple lead times.

    The SIE error is calculated as the signed difference between the predicted and
    true sea ice extent for each forecast day. Sea ice presence is defined by having
    a concentration greater than the threshold value.
    """

    def _batch_error(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds_extent = preds > SEA_ICE_THRESHOLD
        target_extent = target > SEA_ICE_THRESHOLD
        land_mask = getattr(self, "land_mask", None)
        if land_mask is not None:
            preds_extent = preds_extent & land_mask
            target_extent = target_extent & land_mask

        # Calculate the SIE for each day of the forecast
        pred_sie = torch.sum(preds_extent, dim=(2, 3, 4))  # Shape: (B, T)
        true_sie = torch.sum(target_extent, dim=(2, 3, 4))  # Shape: (B, T)
        return (pred_sie - true_sie).float()


class IntegratedIceEdgeErrorPerForecastDay(_MeanIceAreaMetric):
    """Integrated Ice Edge Error (IIEE) metric (in km^2) for use at multiple lead times.

    IIEE is the area of the symmetric difference between the predicted and true ice
    extent: the total area where the two disagree on ice presence, following
    Goessling et al. (2016, https://doi.org/10.1002/2015GL067232). Sea ice presence
    is defined by having a probability greater than the threshold value.

    Unlike `SeaIceExtentErrorPerForecastDay`, which is a signed difference of extents
    and can cancel out over- and under-estimation, IIEE always accumulates
    disagreement and is therefore always >= |SIEError|.
    """

    def _batch_error(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds_extent = preds > SEA_ICE_THRESHOLD
        target_extent = target > SEA_ICE_THRESHOLD
        disagreement = self._masked_mismatch(preds_extent, target_extent)
        return torch.sum(disagreement, dim=(2, 3, 4))


class DistanceAveragedIceEdgeErrorPerForecastDay(_IceAreaMetricBase):
    """Distance-averaged Integrated Ice Edge Error (DIIEE), in km, per lead time.

    The total misclassified-ice area (overestimation + underestimation, i.e. IIEE,
    see `IntegratedIceEdgeErrorPerForecastDay`) is normalised by the combined length
    of the predicted and true ice edges, giving an average displacement distance in
    km: roughly, how far the ice edge would need to move to reconcile the two fields.

        DIIEE = 2 * (over_area + under_area) / (pred_edge_length + true_edge_length)

    Edge length is approximated on the raster grid as
    ``(edge cell count) * pixel_size``; this is coarser than a true vector polygon
    perimeter, and needs no vector geometry, though an optional ``land_mask`` can be
    supplied to exclude land/ice boundaries from the edge count (see `__init__`).
    Lead times where both fields are entirely ice or entirely ice-free (combined edge
    length zero) are undefined and reported as NaN, rather than the ``-9999.99``
    sentinel used upstream, to compose correctly with tensor reductions (e.g.
    `nanmean`).
    """

    def __init__(
        self, pixel_size: int = 25, land_mask: torch.Tensor | None = None
    ) -> None:
        """Initialize the DIIEE metric (see `_IceAreaMetricBase` for parameters)."""
        super().__init__(pixel_size=pixel_size, land_mask=land_mask)
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
        land_mask = getattr(self, "land_mask", None)

        # Combined over- and under-estimation area (the IIEE integral)
        mismatch_map = self._masked_mismatch(preds_extent, target_extent)
        mismatch = mismatch_map.sum(dim=(-2, -1))
        mismatch_area = (
            mismatch.view(batch_size, n_steps, n_channels).sum(dim=(0, 2))
            * self.pixel_size**2
        )

        # Combined predicted + true ice-edge length
        pred_edge_cells = binary_edge(preds_extent, land_mask).sum(dim=(-2, -1)).float()
        target_edge_cells = (
            binary_edge(target_extent, land_mask).sum(dim=(-2, -1)).float()
        )
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
