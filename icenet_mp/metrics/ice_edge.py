"""Shared raster ice-edge detection, used by both FSS and DIIEE."""

import torch
import torch.nn.functional as F


def binary_edge(ice_mask: torch.Tensor) -> torch.Tensor:
    """Boolean ice-edge map: True for ice cells that border a non-ice cell.

    Parameters
    ----------
    ice_mask : torch.Tensor
        Boolean tensor of shape (N, H, W).

    """
    padded = F.pad(ice_mask.float(), (1, 1, 1, 1), value=0.0).bool()
    up = padded[:, :-2, 1:-1]
    down = padded[:, 2:, 1:-1]
    left = padded[:, 1:-1, :-2]
    right = padded[:, 1:-1, 2:]
    return ice_mask & (
        (ice_mask != up) | (ice_mask != down) | (ice_mask != left) | (ice_mask != right)
    )
