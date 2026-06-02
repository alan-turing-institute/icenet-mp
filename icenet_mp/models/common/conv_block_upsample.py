from torch import nn

from icenet_mp.types import TensorNCHW

from .activations import ACTIVATION_FROM_NAME
from .conv_norm_act import ConvNormAct
from .weighted_upsample import WeightedUpsample


class ConvBlockUpsample(nn.Module):
    """Convolutional block that doubles spatial dimensions using two stacked ConvNormAct mini-blocks.

    If out_channels is not specified than this will halve the number of input channels.

    Reverse of ConvBlockDownsample, using upsampling to avoid checkerboarding.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        activation: str = "ReLU",
        kernel_size: int = 3,
        norm_type: str = "batchnorm",
        out_channels: int | None = None,
    ) -> None:
        """Initialize a ConvBlockUpsample module.

        Args:
            in_channels: the number of input channels.
            activation: the activation function to use.
            kernel_size: the size of the convolutional kernel.
            norm_type: type of normalization ("groupnorm", "batchnorm", or "none").
            out_channels: the number of output channels (if None, half of in_channels).

        """
        super().__init__()

        # Set output channels to default if not specified
        out_channels = in_channels // 2 if out_channels is None else out_channels

        self.model = nn.Sequential(
            # Size increasing upsample/normalisation/activation
            WeightedUpsample(in_channels, out_channels=out_channels, upsample_factor=2),
            nn.BatchNorm2d(out_channels),
            ACTIVATION_FROM_NAME[activation](inplace=True),
            # Size preserving convolution/normalisation/activation
            ConvNormAct(
                out_channels,
                out_channels,
                activation=activation,
                kernel_size=kernel_size,
                norm_type=norm_type,
            ),
            # Size preserving convolution/normalisation/activation
            ConvNormAct(
                out_channels,
                out_channels,
                activation=activation,
                kernel_size=kernel_size,
                norm_type=norm_type,
            ),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
