from typing import Any

from torch import nn

from icenet_mp.models.common import (
    CommonConvBlock,
    NormalisedFold,
    Permute,
)
from icenet_mp.types import TensorNCHW

from .base_decoder import BaseDecoder


class PiecewiseDecoder(BaseDecoder):
    """Piecewise decoder that combines data patches from a latent space to build the output space.

    - Initial convolutional block at input resolution
    - Combine patches into output of size output_height x output_width
    - Final convolutional block at output resolution
    - Normalise then bound the output

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)
    """

    def __init__(
        self,
        *,
        conv_activation: str = "SiLU",
        conv_kernel_size: int = 3,
        conv_subblocks_initial: int = 3,
        conv_subblocks_final: int = 3,
        use_hann_window: bool = True,
        use_final_normalisation: bool = True,
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

        # Construct the list of layers
        layers: list[nn.Module] = []
        self.input_channel_indices: tuple[int, ...] | None = None

        if (self.data_space_in.channels != input_channels_required) and (
            conv_subblocks_initial < 1
        ):
            # With no learned channel-reduction convolution, select the requested target
            # variable from each target-group patch explicitly.
            if (
                self.target_channel_offset is None
                or self.target_group_channels is None
                or not self.target_variable_indices
            ):
                msg = (
                    "A convolution-free PiecewiseDecoder needs target latent-channel "
                    "metadata when the combined latent contains additional channels."
                )
                raise ValueError(msg)
            if len(self.target_variable_indices) != self.data_space_out.channels:
                msg = (
                    f"Expected {self.data_space_out.channels} target variable indices, "
                    f"got {len(self.target_variable_indices)}."
                )
                raise ValueError(msg)
            if any(
                index < 0 or index >= self.target_group_channels
                for index in self.target_variable_indices
            ):
                msg = (
                    "target_variable_indices must refer to channels in the target "
                    f"input group with {self.target_group_channels} channel(s)."
                )
                raise ValueError(msg)

            self.input_channel_indices = tuple(
                self.target_channel_offset
                + patch_idx * self.target_group_channels
                + variable_idx
                for patch_idx in range(n_patches)
                for variable_idx in self.target_variable_indices
            )
            if max(self.input_channel_indices) >= self.data_space_in.channels:
                msg = "Target piecewise channel selection exceeds combined latent channels."
                raise ValueError(msg)

        # Optionally add an initial convolutional block at input resolution.
        # This will also set the correct number of channels if needed.
        if conv_subblocks_initial > 0:
            layers.append(
                CommonConvBlock(
                    self.data_space_in.channels,
                    input_channels_required,
                    kernel_size=conv_kernel_size,
                    activation=conv_activation,
                    n_subblocks=conv_subblocks_initial,
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
                use_hann_window=use_hann_window,
            )
        )

        # Optionally add a final convolutional block at output resolution
        if conv_subblocks_final > 0:
            layers.append(
                CommonConvBlock(
                    self.data_space_out.channels,
                    self.data_space_out.channels,
                    kernel_size=conv_kernel_size,
                    activation=conv_activation,
                    n_subblocks=conv_subblocks_final,
                ),
            )

        # Normalise the folded output before bounding it. We set affine=False to avoid
        # saturation that can cause the output to collapse to a constant prediction.
        if use_final_normalisation:
            layers.append(nn.BatchNorm2d(self.data_space_out.channels, affine=False))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space by combining patches.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        if self.input_channel_indices is not None:
            x = x[:, self.input_channel_indices, :, :]
        return self.model(x)
