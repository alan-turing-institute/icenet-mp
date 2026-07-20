"""Downsample block wrapped in DC-AE's parameter-free "Residual Autoencoding" shortcut."""

from typing import Any

from torch import Tensor, nn

from .channel_adaptor import ChannelAdaptor


class ResidualDownsample(nn.Module):
    """A parametric downsample plus a non-parametric residual shortcut.

    Mirror of :class:`ResidualUpsample` for the encoder side.

    The parametric block is either PixelUnshuffle plus Conv2d or strided Conv2d.
    The shortcut block is a non-parametric operation that matches the type of spatial
    downsampling used in the parametric block (PixelUnshuffle or Pool) plus a channel
    adaptor that matches the number of output channels.

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
        """Initialise a ResidualDownsample.

        Args:
            in_channels: the number of input channels.
            out_channels: the number of output channels.
            factor: the spatial downsampling factor.
            pixel_shuffle: downsample with PixelUnshuffle if True, else strided convolution.
            **kwargs: forwarded to the parametric block convolutional layer

        """
        super().__init__()
        if pixel_shuffle:
            self.parametric = nn.Sequential(
                nn.PixelUnshuffle(factor),
                nn.Conv2d(in_channels * factor**2, out_channels, **kwargs),
            )
            self.shortcut = nn.Sequential(
                nn.PixelUnshuffle(factor),
                ChannelAdaptor(in_channels * factor**2, out_channels),
            )
        else:
            self.parametric = nn.Conv2d(
                in_channels, out_channels, stride=factor, **kwargs
            )
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(factor), ChannelAdaptor(in_channels, out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the parametric block plus its residual shortcut to x."""
        return self.parametric(x) + self.shortcut(x)
