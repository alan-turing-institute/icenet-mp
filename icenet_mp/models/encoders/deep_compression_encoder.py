"""Deep Compression encoder following the Deep Compression AutoEncoder architecture.

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
    Patchify2D,
    ResBlock,
    make_conv2d,
)
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
        """Initialise a DeepCompressionEncoder."""
        super().__init__(**kwargs)

        if len(hid_blocks) != len(hid_channels):
            msg = f"hid_blocks and hid_channels must have the same length, got {len(hid_blocks)} and {len(hid_channels)}"
            raise ValueError(msg)
        in_channels = self.data_space_in.channels

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

        self.patch: nn.Module = (
            Patchify2D(patch_size) if math.prod(patch_size) > 1 else nn.Identity()
        )

        self.descent = nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            blocks: list[nn.Module] = []

            if i == 0:
                # First level: project from (possibly patchified) input channels
                blocks.append(
                    make_conv2d(
                        math.prod(patch_size) * in_channels,
                        hid_channels[i],
                        **conv_kwargs,
                    )
                )
            elif pixel_shuffle:
                # Downsample via patchify then project
                blocks.append(
                    nn.Sequential(
                        Patchify2D(stride),
                        make_conv2d(
                            hid_channels[i - 1] * math.prod(stride),
                            hid_channels[i],
                            identity_init=identity_init,
                            **conv_kwargs,
                        ),
                    )
                )
            else:
                # Downsample via strided convolution
                blocks.append(
                    make_conv2d(
                        hid_channels[i - 1],
                        hid_channels[i],
                        identity_init=identity_init,
                        stride=stride,
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
                for _ in range(num_blocks)
            )

            if i + 1 == len(hid_blocks):
                # Final level: project to latent channels
                blocks.append(
                    make_conv2d(
                        hid_channels[i],
                        latent_channels,
                        identity_init=identity_init,
                        **conv_kwargs,
                    )
                )

            self.descent.append(nn.ModuleList(blocks))

        self.data_space_out.channels = latent_channels

        logger.debug(
            "DCEncoder (%s): %d levels, %d -> %d channels",
            self.name,
            len(hid_channels),
            in_channels,
            latent_channels,
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Encode a single timestep from input space to latent space.

        Args:
            x: TensorNCHW with (batch_size, input_channels, H, W)

        Returns:
            TensorNCHW with (batch_size, latent_channels, H_lat, W_lat)

        """
        x = self.patch(x)
        for blocks in self.descent:
            for block in blocks:
                x = block(x)
        return x
