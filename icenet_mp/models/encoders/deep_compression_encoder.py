"""Deep Compression encoder following the Deep Compression AutoEncoder architecture.

Reference:
    Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
    (Chen et al., 2024) [https://arxiv.org/abs/2410.10733]
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal

from torch import nn

from icenet_mp.models.common import ResBlock, ResidualDownsample
from icenet_mp.types import TensorNCHW

from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)


class DeepCompressionEncoder(BaseEncoder):
    """Encoder following the Deep Compression AutoEncoder (DCAE) architecture.

    Mirror of :class:`DeepCompressionDecoder`.

    - (optional) initial patchify (PixelUnshuffle or Conv2d) step
    - `len(hid_blocks) - 1` layers of downsample (pixel-unshuffle or strided-conv) then `hid_blocks[i]` ResBlocks
    - final convolution to latent channels

    Input space:
        TensorNTCHW with (batch_size, n_timeslices, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, latent_channels, latent_height, latent_width)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        attention_heads: dict[int, int] = {},  # noqa: B006
        attention_scales: tuple[int, ...] = (5,),
        ffn_factor: int = 1,
        hid_blocks: Sequence[int] = (3, 3, 3),
        hid_channels: Sequence[int] = (64, 128, 256),
        kernel_size: int = 3,
        latent_channels: int | None = None,
        norm: str = "groupnorm",
        patch_size: int = 1,
        periodic: bool = False,
        pixel_shuffle: bool = True,
        stride: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialise a DeepCompressionEncoder."""
        super().__init__(**kwargs)

        if len(hid_blocks) != len(hid_channels):
            msg = f"hid_blocks and hid_channels must have the same length, got {len(hid_blocks)} and {len(hid_channels)}"
            raise ValueError(msg)
        if patch_size < 1:
            msg = f"patch_size must be >= 1, got {patch_size}"
            raise ValueError(msg)
        if stride < 1:
            msg = f"stride must be >= 1, got {stride}"
            raise ValueError(msg)
        in_channels = self.data_space_in.channels

        # Validate the output shape is correct.
        spatial_factor = patch_size * stride ** (len(hid_channels) - 1)
        output_shape = (
            self.data_space_in.shape[0] // spatial_factor,
            self.data_space_in.shape[1] // spatial_factor,
        )
        if output_shape != self.data_space_out.shape:
            msg = (
                f"Stride {stride} and number of layers {len(hid_channels)} will encode "
                f"inputs of shape {self.data_space_in.shape} to shape {output_shape} "
                f"but the required latent space shape is {self.data_space_out.shape}"
            )
            raise ValueError(msg)

        # Set latent channels to the last hidden channel if not specified
        latent_channels = latent_channels or hid_channels[-1]

        # Set padding and padding mode for convolutions
        padding = kernel_size // 2
        padding_mode: Literal["circular", "zeros"] = "circular" if periodic else "zeros"

        # Construct list of layers
        layers: list[nn.Module] = []
        logger.debug(
            "DeepCompressionEncoder (%s): %d layers, %d -> %d channels",
            self.name,
            len(hid_channels),
            in_channels,
            latent_channels,
        )

        for idx, num_blocks in enumerate(hid_blocks):
            if idx == 0:
                # Shallowest layer: (optionally patchify) then convolve the input
                layers.append(
                    ResidualDownsample(
                        in_channels=in_channels,
                        out_channels=hid_channels[idx],
                        factor=patch_size,
                        pixel_shuffle=pixel_shuffle if patch_size > 1 else False,
                        kernel_size=kernel_size,
                        padding_mode=padding_mode,
                        padding=padding,
                    )
                )
            else:
                # Subsequent layers: downsample by factor `stride`
                layers.append(
                    ResidualDownsample(
                        in_channels=hid_channels[idx - 1],
                        out_channels=hid_channels[idx],
                        factor=stride,
                        pixel_shuffle=pixel_shuffle,
                        kernel_size=kernel_size,
                        padding_mode=padding_mode,
                        padding=padding,
                    )
                )

            # Add `num_blocks` residual blocks
            layers.extend(
                ResBlock(
                    hid_channels[idx],
                    attention_heads=attention_heads.get(idx),
                    attention_scales=attention_scales,
                    ffn_factor=ffn_factor,
                    kernel_size=kernel_size,
                    norm=norm,
                    padding_mode=padding_mode,
                    padding=padding,
                )
                for _ in range(num_blocks)
            )

            if idx + 1 == len(hid_blocks):
                # Deepest layer: convolve to latent channels.
                layers.append(
                    ResidualDownsample(
                        in_channels=hid_channels[idx],
                        out_channels=latent_channels,
                        factor=1,
                        pixel_shuffle=False,
                        kernel_size=kernel_size,
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                )

        # Set the number of output channels correctly
        self.data_space_out.channels = latent_channels

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space with a DCAE encoder.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
