from torch import nn

from icenet_mp.types import TensorNCHW

from .conv_norm_act import ConvNormAct


class ConvBlockDownsample(nn.Module):
    """Convolutional block that halves spatial dimensions.

    ConvNormAct > ConvNormAct

    If out_channels is not specified then this will double the number of input channels.

    Reverse of ConvBlockUpsample.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        activation: str = "ReLU",
        kernel_size: int = 3,
        n_subblocks: int = 2,
        norm_type: str = "batchnorm",
        out_channels: int | None = None,
    ) -> None:
        """Initialize a ConvBlockDownsample module.

        Args:
            in_channels: the number of input channels.
            activation: the activation function to use.
            kernel_size: the size of the convolutional kernel.
            n_subblocks: the number of ConvNormAct blocks to stack (default 2).
            norm_type: type of normalization ("groupnorm", "batchnorm", or "none").
            out_channels: the number of output channels (if None, double the input channels).

        """
        super().__init__()

        if n_subblocks < 1:
            msg = f"n_subblocks must be at least 1, got {n_subblocks}."
            raise ValueError(msg)

        out_channels = in_channels * 2 if out_channels is None else out_channels
        self.model = nn.Sequential(
            # Size reducing convolution/normalisation/activation that changes channels
            ConvNormAct(
                in_channels,
                out_channels,
                activation=activation,
                kernel_size=kernel_size,
                norm_type=norm_type,
                padding=(kernel_size - 1) // 2,
                stride=2,
            ),
            *(
                # Size preserving convolution/normalisation/activation that maintains channels
                ConvNormAct(
                    out_channels,
                    out_channels,
                    activation=activation,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for _ in range(n_subblocks - 1)
            ),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
