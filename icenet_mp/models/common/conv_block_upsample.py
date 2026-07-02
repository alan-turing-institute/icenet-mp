from torch import nn

from icenet_mp.types import TensorNCHW

from .activations import ACTIVATION_FROM_NAME
from .conv_norm_act import ConvNormAct
from .normalisations import normalisation_from_name
from .weighted_upsample import WeightedUpsample


class ConvBlockUpsample(nn.Module):
    """Convolutional block that doubles spatial dimensions.

    (Upsample > Norm > Act) > ConvNormAct > ConvNormAct

    If out_channels is not specified then this will halve the number of input channels.

    Reverse of ConvBlockDownsample, using upsampling to increase spatial dimensions.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        *,
        activation: str = "ReLU",
        kernel_size: int = 3,
        n_subblocks: int = 2,
        norm_type: str = "batchnorm",
        out_channels: int | None = None,
        upsample_mode: str = "bilinear",
    ) -> None:
        """Initialize a ConvBlockUpsample module.

        Args:
            in_channels: the number of input channels.
            activation: the activation function to use.
            kernel_size: the size of the convolutional kernel.
            n_subblocks: the number of ConvNormAct blocks to stack (default 2).
            norm_type: type of normalization ("groupnorm", "batchnorm", or "none").
            out_channels: the number of output channels (if None, half of in_channels).
            upsample_mode: the method to use for upsampling ("bilinear" or "shuffle").

        """
        super().__init__()

        out_channels = in_channels // 2 if out_channels is None else out_channels

        if n_subblocks < 1:
            msg = f"n_subblocks must be at least 1, got {n_subblocks}."
            raise ValueError(msg)

        if upsample_mode not in ("bilinear", "shuffle"):
            msg = f"Unsupported upsample_mode: {upsample_mode}"
            raise ValueError(msg)

        self.model = nn.Sequential(
            # Upsampling layer to increase spatial dimensions
            {
                "bilinear": nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                ),
                "shuffle": WeightedUpsample(in_channels, upsample_factor=2),
            }[upsample_mode],
            normalisation_from_name(norm_type, in_channels),
            ACTIVATION_FROM_NAME[activation](),
            *(
                # Size preserving convolution/normalisation/activation
                # Final block also changes channels to out_channels
                ConvNormAct(
                    in_channels,
                    out_channels if idx_subblock == n_subblocks - 1 else in_channels,
                    activation=activation,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for idx_subblock in range(n_subblocks)
            ),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
