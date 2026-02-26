from typing import Any

from torch import nn

from icenet_mp.models.common import (
    CommonConvBlock,
    NormalisedFold,
    Permute,
    RestrictRange,
    Shift,
)
from icenet_mp.types import RangeRestriction, TensorNCHW

from .base_decoder import BaseDecoder


class PiecewiseDecoder(BaseDecoder):
    """Piecewise decoder that combines data patches from a latent space to build the output space.

    - 1 convolutional block to set the required number of channels
    - n_conv_blocks of constant-size convolutional blocks
    - Combine patches into output of size output_height x output_width

    Latent space:
        TensorNTCHW with (batch_size, n_forecast_steps, latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
    """

    def __init__(
        self,
        *,
        conv_activation: str = "SiLU",
        conv_kernel_size: int = 3,
        n_conv_blocks: int = 0,
        restrict_range: str = "none",
        **kwargs: Any,
    ) -> None:
        """Initialise a PiecewiseDecoder."""
        super().__init__(**kwargs)

        # Calculate the number of patches required
        # We set the stride to be half the patch size to ensure overlap, which will
        # capture more of the spatial structure of the data.
        strides = tuple(
            max(1, patch_size // 2) for patch_size in self.data_space_in.shape
        )
        n_patches = (
            (
                self.data_space_out.shape[0]
                + 2 * strides[0]
                - 1 * (self.data_space_in.shape[0] - 1)
                - 1
            )
            // strides[0]
            + 1
        ) * (
            (
                self.data_space_out.shape[1]
                + 2 * strides[1]
                - 1 * (self.data_space_in.shape[1] - 1)
                - 1
            )
            // strides[1]
            + 1
        )
        input_channels_required = self.data_space_out.channels * n_patches

        # Construct list of layers
        layers: list[nn.Module] = []

        # If necessary, add a convolutional block to get the required number of channels
        if (n_conv_blocks != 0) or (
            self.data_space_in.channels != input_channels_required
        ):
            layers.append(
                CommonConvBlock(
                    self.data_space_in.channels,
                    input_channels_required,
                    kernel_size=conv_kernel_size,
                    activation=conv_activation,
                    n_subblocks=n_conv_blocks + 1,
                ),
            )

        # Unflatten the channel dimension to extract the patches: [N, n_patches, C, patch_h, patch_w]
        layers.append(nn.Unflatten(1, (n_patches, -1)))

        # Flatten the patch dimensions: [N, n_patches, C * patch_area]
        layers.append(nn.Flatten(2, 4))

        # Permute dimensions: [N, C * patch_area, n_patches]
        layers.append(Permute((0, 2, 1)))

        # Fold patches into the output shape: [N, C, output_h, output_w]
        layers.append(
            NormalisedFold(
                output_size=self.data_space_out.shape,
                kernel_size=self.data_space_in.shape,
                stride=strides,
                padding=strides,
            )
        )

        # Apply a scale and offset shift to reduce the large values caused by folding
        # multiple pixels into a single output pixel.
        layers.append(Shift(scale=True, offset=True))

        # Specify how/whether the output is bounded between 0 and 1
        if (method := RangeRestriction(restrict_range)) != RangeRestriction.NONE:
            layers.append(RestrictRange(method, min_val=0, max_val=1))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space by combining patches.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        return self.model(x)
