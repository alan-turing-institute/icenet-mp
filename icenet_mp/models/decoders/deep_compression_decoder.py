"""Deep Compression decoder following the Deep Compression AutoEncoder architecture.

Reference:
    Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
    (Chen et al., 2024)
"""

import logging
import math
from collections.abc import Sequence
from typing import Any

from torch import nn

from icenet_mp.models.common.dcae_blocks import (
    ResBlock,
    Unpatchify2D,
)
from icenet_mp.types import TensorNCHW

from .base_decoder import BaseDecoder

logger = logging.getLogger(__name__)


class DeepCompressionDecoder(BaseDecoder):
    """Decoder following the Deep Compression AutoEncoder (DCAE) architecture.

    Mirror of :class:`DeepCompressionEncoder`: ascends from latent channels through
    levels of ResBlocks with pixel-shuffle (or nearest-upsample) upsampling between
    levels, ending with an optional unpatchify step.

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
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 2,
        patch_size: int | tuple[int, int] = 1,
        pixel_shuffle: bool = True,
        norm: str = "group",
        groups: int = 16,
        attention_heads: dict[int, int] = {},  # noqa: B006
        ffn_factor: int = 1,
        periodic: bool = False,
        dropout: float | None = None,
        checkpointing: bool = False,
        identity_init: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialise a DeepCompressionDecoder."""
        super().__init__(**kwargs)

        if len(hid_blocks) != len(hid_channels):
            msg = f"hid_blocks and hid_channels must have the same length, got {len(hid_blocks)} and {len(hid_channels)}"
            raise ValueError(msg)
        in_channels = self.data_space_in.channels
        out_channels = self.data_space_out.channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        conv_kwargs = {
            "kernel_size": kernel_size,
            "padding": (kernel_size[0] // 2, kernel_size[1] // 2),
            "padding_mode": "circular" if periodic else "zeros",
        }

        self.unpatch: nn.Module = (
            Unpatchify2D(patch_size) if math.prod(patch_size) > 1 else nn.Identity()
        )

        self.ascent = nn.ModuleList()

        # Build levels in reverse order (deepest first) to mirror the encoder
        for i in reversed(range(len(hid_blocks))):
            blocks: list[nn.Module] = []

            if i + 1 == len(hid_blocks):
                # Deepest level: project from latent channels
                blocks.append(
                    nn.Conv2d(
                        in_channels,
                        hid_channels[i],
                        **conv_kwargs,
                    )
                )

            blocks.extend(
                ResBlock(
                    hid_channels[i],
                    norm=norm,
                    groups=groups,
                    attention_heads=attention_heads.get(i),
                    ffn_factor=ffn_factor,
                    dropout=dropout,
                    checkpointing=checkpointing,
                    **conv_kwargs,
                )
                for _ in range(hid_blocks[i])
            )

            if i > 0:
                if pixel_shuffle:
                    # Upsample via project then unpatchify
                    blocks.append(
                        nn.Sequential(
                            nn.Conv2d(
                                hid_channels[i],
                                hid_channels[i - 1] * math.prod(stride),
                                **conv_kwargs,
                            ),
                            Unpatchify2D(stride),
                        )
                    )
                else:
                    # Upsample via nearest-neighbour then project
                    blocks.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=stride, mode="nearest"),
                            nn.Conv2d(
                                hid_channels[i],
                                hid_channels[i - 1],
                                **conv_kwargs,
                            ),
                        )
                    )
            else:
                # Shallowest level: project to (possibly pre-unpatchify) output channels
                blocks.append(
                    nn.Conv2d(
                        hid_channels[i],
                        math.prod(patch_size) * out_channels,
                        **conv_kwargs,
                    )
                )

            self.ascent.append(nn.ModuleList(blocks))

        logger.debug(
            "DCDecoder (%s): %d levels, %d → %d channels",
            self.name,
            len(hid_channels),
            in_channels,
            out_channels,
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Decode a single timestep from latent space to output space.

        Args:
            x: TensorNCHW with (batch_size, latent_channels, H_lat, W_lat)

        Returns:
            TensorNCHW with (batch_size, output_channels, H, W)

        """
        for blocks in self.ascent:
            for block in blocks:
                x = block(x)
        return self.unpatch(x)
