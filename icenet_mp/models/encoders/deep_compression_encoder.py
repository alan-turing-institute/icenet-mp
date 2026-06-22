"""Deep Compression encoder following the Deep Compression AutoEncoder architecture.

Reference:
    Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
    (Chen et al., 2024) [https://arxiv.org/abs/2410.10733]
"""

import logging
from collections.abc import Sequence
from typing import Any

from torch import nn

from icenet_mp.models.common.dcae_blocks import ResBlock
from icenet_mp.types import TensorNCHW

from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)


class DeepCompressionEncoder(BaseEncoder):
    """Encoder following the Deep Compression AutoEncoder (DCAE) architecture.

    Applies an optional initial patchify step, then descends through levels of
    ResBlocks with pixel-shuffle (or strided-conv) downsampling between levels.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, H, W)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, latent_channels, H/s^D, W/s^D)
        where s is ``stride`` and D is ``len(hid_channels)``.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        latent_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: int = 3,
        stride: int = 2,
        patch_size: int = 1,
        pixel_shuffle: bool = True,
        norm: str = "groupnorm",
        attention_heads: dict[int, int] = {},  # noqa: B006
        ffn_factor: int = 1,
        periodic: bool = False,
        dropout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise a DeepCompressionEncoder."""
        super().__init__(**kwargs)

        if len(hid_blocks) != len(hid_channels):
            msg = f"hid_blocks and hid_channels must have the same length, got {len(hid_blocks)} and {len(hid_channels)}"
            raise ValueError(msg)
        in_channels = self.data_space_in.channels

        conv_kwargs = {
            "kernel_size": kernel_size,
            "padding": kernel_size // 2,
            "padding_mode": "circular" if periodic else "zeros",
        }

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
                if patch_size > 1:
                    layers.append(nn.PixelUnshuffle(patch_size))
                layers.append(
                    nn.Conv2d(
                        patch_size**2 * in_channels,
                        hid_channels[idx],
                        **conv_kwargs,
                    )
                )
            elif pixel_shuffle:
                # Subsequent layers: downsample via patchify then convolve
                layers.extend(
                    (
                        nn.PixelUnshuffle(stride),
                        nn.Conv2d(
                            hid_channels[idx - 1] * stride**2,
                            hid_channels[idx],
                            **conv_kwargs,
                        ),
                    )
                )
            else:
                # Subsequent layers: downsample via strided convolution
                layers.append(
                    nn.Conv2d(
                        hid_channels[idx - 1],
                        hid_channels[idx],
                        stride=stride,
                        **conv_kwargs,
                    )
                )

            # Add `num_blocks` residual blocks
            layers.extend(
                ResBlock(
                    hid_channels[idx],
                    norm=norm,
                    attention_heads=attention_heads.get(idx),
                    ffn_factor=ffn_factor,
                    dropout=dropout,
                    **conv_kwargs,
                )
                for _ in range(num_blocks)
            )

            if idx + 1 == len(hid_blocks):
                # Deepest layer: convolve to latent channels
                layers.append(
                    nn.Conv2d(
                        hid_channels[idx],
                        latent_channels,
                        **conv_kwargs,
                    )
                )

        # Set the number of output channels correctly
        self.data_space_out.channels = latent_channels

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Encode a single timestep from input space to latent space.

        Args:
            x: TensorNCHW with (batch_size, input_channels, H, W)

        Returns:
            TensorNCHW with (batch_size, latent_channels, H_lat, W_lat)

        """
        return self.model(x)
