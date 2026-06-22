"""Deep Compression decoder following the Deep Compression AutoEncoder architecture.

Reference:
    Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
    (Chen et al., 2024)
"""

import logging
from collections.abc import Sequence
from typing import Any

from torch import nn

from icenet_mp.models.common import ResBlock
from icenet_mp.types import TensorNCHW

from .base_decoder import BaseDecoder

logger = logging.getLogger(__name__)


class DeepCompressionDecoder(BaseDecoder):
    """Decoder following the Deep Compression AutoEncoder (DCAE) architecture.

    Mirror of :class:`DeepCompressionEncoder`: ascends from latent channels through
    layers of ResBlocks with pixel-shuffle (or nearest-upsample) upsampling between
    layers, ending with an optional unpatchify step.

    Latent space:
        TensorNTCHW with (batch_size, n_forecast_steps, latent_channels, H_lat, W_lat)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, output_channels, H, W)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
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
        """Initialise a DeepCompressionDecoder."""
        super().__init__(**kwargs)

        if len(hid_blocks) != len(hid_channels):
            msg = f"hid_blocks and hid_channels must have the same length, got {len(hid_blocks)} and {len(hid_channels)}"
            raise ValueError(msg)
        in_channels = self.data_space_in.channels
        out_channels = self.data_space_out.channels

        conv_kwargs = {
            "kernel_size": kernel_size,
            "padding": kernel_size // 2,
            "padding_mode": "circular" if periodic else "zeros",
        }

        # Construct list of layers
        layers: list[nn.Module] = []
        logger.debug(
            "DeepCompressionDecoder (%s): %d layers, %d -> %d channels",
            self.name,
            len(hid_channels),
            in_channels,
            out_channels,
        )

        # Build in reverse order (deepest first) to mirror the encoder
        for idx in reversed(range(len(hid_blocks))):
            if idx + 1 == len(hid_blocks):
                # Deepest layer: convolve from latent channels
                layers.append(nn.Conv2d(in_channels, hid_channels[idx], **conv_kwargs))

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
                for _ in range(hid_blocks[idx])
            )

            if idx > 0:
                if pixel_shuffle:
                    # Subsequent layers: upsample via convolve then unpatchify
                    layers.extend(
                        (
                            nn.Conv2d(
                                hid_channels[idx],
                                hid_channels[idx - 1] * stride**2,
                                **conv_kwargs,
                            ),
                            nn.PixelShuffle(stride),
                        )
                    )
                else:
                    # Subsequent layers: upsample via interpolation then convolve
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=stride, mode="nearest"),
                            nn.Conv2d(
                                hid_channels[idx], hid_channels[idx - 1], **conv_kwargs
                            ),
                        )
                    )
            else:
                # Shallowest layer: convolve then (optionally unpatchify)
                layers.append(
                    nn.Conv2d(
                        hid_channels[idx], patch_size**2 * out_channels, **conv_kwargs
                    )
                )
                if patch_size > 1:
                    layers.append(nn.PixelShuffle(patch_size))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Decode a single timestep from latent space to output space.

        Args:
            x: TensorNCHW with (batch_size, latent_channels, H_lat, W_lat)

        Returns:
            TensorNCHW with (batch_size, output_channels, H, W)

        """
        return self.model(x)
