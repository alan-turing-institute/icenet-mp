from typing import Any

from torch import nn

from icenet_mp.models.common import CommonConvBlock, Permute
from icenet_mp.types import TensorNCHW

from .base_encoder import BaseEncoder


class PiecewiseEncoder(BaseEncoder):
    """Piecewise encoder that splits data from an input space into smaller patches in a latent space.

    - Split into patches of size latent_height x latent_width
    - n_conv_blocks of constant-size convolutional blocks

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        conv_activation: str = "SiLU",
        conv_kernel_size: int = 3,
        n_conv_blocks: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialise a PiecewiseEncoder."""
        super().__init__(**kwargs)

        # Calculate the number of patches required
        # We set the stride to be half the patch size to ensure overlap, which will
        # capture more of the spatial structure of the data.
        strides = tuple(
            max(1, patch_size // 2) for patch_size in self.data_space_out.shape
        )
        n_patches = (
            (
                self.data_space_in.shape[0]
                + 2 * strides[0]
                - 1 * (self.data_space_out.shape[0] - 1)
                - 1
            )
            // strides[0]
            + 1
        ) * (
            (
                self.data_space_in.shape[1]
                + 2 * strides[1]
                - 1 * (self.data_space_out.shape[1] - 1)
                - 1
            )
            // strides[1]
            + 1
        )

        # Set the number of output channels correctly
        self.data_space_out.channels = self.data_space_in.channels * n_patches

        # Construct the list of layers
        layers = [
            # Unfold into patches of size data_space_out.shape: [N, C * patch_area, n_patches]
            nn.Unfold(
                self.data_space_out.shape,
                stride=strides,
                padding=strides,
            ),
            # Permute dimensions: [N, n_patches, C * patch_area]
            Permute((0, 2, 1)),
            # Unflatten the patches: [N, n_patches, C, patch_h, patch_w]
            nn.Unflatten(2, (-1, *self.data_space_out.shape)),
            # Combine into a single channel dimension: [N, C * n_patches, patch_h, patch_w]
            nn.Flatten(1, 2),
        ]

        # Optionally add non-linearity with convolutional blocks
        if n_conv_blocks > 0:
            layers.append(
                CommonConvBlock(
                    self.data_space_out.channels,
                    self.data_space_out.channels,
                    kernel_size=conv_kernel_size,
                    activation=conv_activation,
                    n_subblocks=n_conv_blocks,
                ),
            )

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space by splitting into patches.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
