"""Shared raster ice-edge detection, used by both FSS and DIIEE."""

import torch
import torch.nn.functional as F


def binary_edge(
    ice_mask: torch.Tensor, land_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Boolean ice-edge map: True for ice cells that border a non-ice ocean cell.

    Parameters
    ----------
    ice_mask : torch.Tensor
        Boolean tensor of shape (N, H, W).
    land_mask : torch.Tensor, optional
        Boolean tensor of shape (H, W), True for ocean cells and False for land.
        When given, land cells are excluded from the edge test — both as neighbors
        (an ice cell bordering only land is not counted as an edge cell, though it
        is still counted if it also borders true open water) and from the returned
        map itself, so a land cell is never reported as an edge cell even if its raw
        (unmasked) value happens to read as "ice".

    Cells beyond the grid boundary are treated as matching the cell they border (via
    replicate padding), rather than being manufactured as non-ice: the domain's own
    edge is not itself an ice/ocean transition, so it should not be able to invent a
    disagreement just because a real edge would need one more ring of pixels to
    resolve. A genuine ice edge that runs along the domain boundary is still detected
    normally, since it disagrees with its interior (in-grid) neighbors regardless.

    """
    comparison_mask = ice_mask
    if land_mask is not None:
        comparison_mask = ice_mask | ~land_mask.bool()
    padded = F.pad(comparison_mask.float(), (1, 1, 1, 1), mode="replicate").bool()
    up = padded[:, :-2, 1:-1]
    down = padded[:, 2:, 1:-1]
    left = padded[:, 1:-1, :-2]
    right = padded[:, 1:-1, 2:]
    edge = ice_mask & (
        (ice_mask != up) | (ice_mask != down) | (ice_mask != left) | (ice_mask != right)
    )
    if land_mask is not None:
        edge = edge & land_mask.bool()
    return edge
