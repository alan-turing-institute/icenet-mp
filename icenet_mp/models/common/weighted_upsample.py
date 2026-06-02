import torch
from torch import Tensor, nn
from torch.nn.init import kaiming_normal_


class WeightedUpsample(nn.Module):
    """Learned spatial upsampling via sub-pixel convolution (PixelShuffle).

    Conv2d > PixelShuffle

    The initial Conv2d is ICNR-initialised so all sub-channel groups
    start identical, preventing early-training checkerboard.
    """

    def __init__(self, channels: int, *, upsample_factor: int = 2) -> None:
        """Initialise a WeightedUpsample module.

        Args:
            channels: the number of channels.
            upsample_factor: the spatial upsampling factor.

        """
        super().__init__()

        # Initial convolution to produce the required channels for PixelShuffle
        # We apply a kernel of the same size as the upsampling factor to allow spatial
        # mixing at the same scale.
        initial_conv = nn.Conv2d(
            channels,
            channels * upsample_factor**2,
            kernel_size=upsample_factor,
            padding="same",
        )

        # ICNR initialisation from https://arxiv.org/abs/1707.02937.
        # Set all sub-channel groups to a common Kaiming-normal kernel so the initial
        # state for PixelShuffle looks like nearest-neighbour upsampling.
        group_weights = initial_conv.weight.new_empty(
            channels, *initial_conv.weight.shape[1:]
        )
        kaiming_normal_(group_weights)
        with torch.no_grad():
            initial_conv.weight.copy_(
                group_weights.repeat_interleave(upsample_factor**2, dim=0)
            )

        self.block = nn.Sequential(
            # Convolution: [N, C, H, W] -> [N, C * r^2, H, W]
            initial_conv,
            # PixelShuffle: [N, C * r^2, H, W] -> [N, C, H * r, W * r]
            nn.PixelShuffle(upsample_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
