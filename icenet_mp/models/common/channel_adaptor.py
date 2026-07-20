"""Parameter-free channel-count change: shrink via grouped averaging, grow via nearest duplication."""

import torch.nn.functional as F  # noqa: N812
from torch import nn

from icenet_mp.types import TensorNCHW


class ChannelAdaptor(nn.Module):
    """Deterministically change the number of channels of an NCHW tensor.

    Reducing the number of channels averages across channels, while increasing simply
    duplicates channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialise a ChannelAdapt."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Apply the channel adaptation to x."""
        if self.out_channels == self.in_channels:
            return x

        # Treat the channel axis as the "depth" dimension of a 3D tensor so a single
        # 3D pooling/interpolation call adapts channels while leaving H, W untouched.
        _, _, h, w = x.shape
        y = x.unsqueeze(1)  # (B, 1, C_in, H, W)
        if self.out_channels < self.in_channels:
            y = F.adaptive_avg_pool3d(y, (self.out_channels, h, w))
        else:
            y = F.interpolate(y, size=(self.out_channels, h, w), mode="nearest")
        return y.squeeze(1)
