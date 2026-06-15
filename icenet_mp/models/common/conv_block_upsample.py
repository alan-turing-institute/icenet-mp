from torch import nn

from icenet_mp.types import TensorNCHW

from .activations import ACTIVATION_FROM_NAME
from .conv_norm_act import ConvNormAct
from .normalisations import normalisation_from_name
from .weighted_upsample import WeightedUpsample


class ConvBlockUpsample(nn.Module):
    """Convolutional block that doubles spatial dimensions.

    (WeightedUpsample > Norm > Act) > ConvNormAct > ConvNormAct

    If out_channels is not specified then this will halve the number of input channels.

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

        out_channels = in_channels // 2 if out_channels is None else out_channels

        self.model = nn.Sequential(
            # Size increasing upsample/normalisation/activation that maintains channels
            WeightedUpsample(in_channels, upsample_factor=2),
            normalisation_from_name(norm_type, in_channels),
            ACTIVATION_FROM_NAME[activation](inplace=True),
            # Size preserving convolution/normalisation/activation that maintains channels
            ConvNormAct(
                in_channels,
                in_channels,
                activation=activation,
                kernel_size=kernel_size,
                norm_type=norm_type,
            ),
            # Size preserving convolution/normalisation/activation that changes channels
            ConvNormAct(
                in_channels,
                out_channels,
                activation=activation,
                kernel_size=kernel_size,
                norm_type=norm_type,
            ),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
