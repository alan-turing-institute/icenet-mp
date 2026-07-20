"""Upsample block wrapped in DC-AE's parameter-free "Residual Autoencoding" shortcut."""

from typing import Any

from torch import Tensor, nn

from .channel_adaptor import ChannelAdaptor
from .weighted_upsample import WeightedUpsample


class ResidualUpsample(nn.Module):
    """A parametric upsample plus a non-parametric residual shortcut.

    Mirror of :class:`ResidualDownsample` for the decoder side.

    The parametric block is either WeightedUpsample or Upsample + Conv2d.
    The shortcut block is a channel adaptor that matches the number of output channels.
    plus a non-parametric operation that matches the type of spatial upsampling used in
    the parametric block (PixelShuffle or Upsample).

    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        factor: int,
        pixel_shuffle: bool,
        **kwargs: Any,
    ) -> None:
        """Initialise a ResidualUpsample.

        Args:
            in_channels: the number of input channels.
            out_channels: the number of output channels.
            factor: the spatial upsampling factor.
            pixel_shuffle: upsample with PixelShuffle if True, else strided convolution.
            **kwargs: forwarded to the parametric block convolutional layer

        """
        super().__init__()
        if pixel_shuffle:
            self.parametric = WeightedUpsample(
                in_channels, out_channels=out_channels, upsample_factor=factor
            )
            self.shortcut = nn.Sequential(
                ChannelAdaptor(in_channels, out_channels * factor**2),
                nn.PixelShuffle(factor),
            )
        else:
            self.parametric = nn.Sequential(
                nn.Upsample(scale_factor=factor, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, **kwargs),
            )
            self.shortcut = nn.Sequential(
                ChannelAdaptor(in_channels, out_channels),
                nn.Upsample(scale_factor=factor, mode="nearest"),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the parametric block plus its residual shortcut to x."""
        return self.parametric(x) + self.shortcut(x)
