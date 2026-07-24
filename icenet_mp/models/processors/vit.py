"""Vision Transformer implementation.

Description:
    Vision Transformer (ViT) model for sea ice forecasting that predicts future sea ice
    concentration from meteorological data. Takes multi-channel input images, converts
    them to patch embeddings, processes through transformer encoder blocks, and outputs
    spatially-resolved predictions for specified forecast horizons.
"""

from typing import Any

import torch
from torch import nn

from icenet_mp.models.common import (
    CommonConvBlock,
    PatchEmbedding,
    TransformerEncoderBlock,
)
from icenet_mp.types import TensorNCHW

from .base_processor import BaseProcessor


class VitProcessor(BaseProcessor):
    def __init__(  # noqa: PLR0913
        self,
        *,
        depth: int = 3,
        dropout: float = 0.3,
        emb_dim: int = 128,
        heads: int = 4,
        mlp_dim: int = 256,
        patch_size: int = 16,
        decode_head: str = "linear",
        refine_channels: int = 64,
        refine_kernel_size: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize Vision Transformer model for sea ice forecasting.

        ``decode_head`` selects how patch tokens are turned back into an output map.
        "linear" (default) reproduces the original per-patch linear decode plus a
        single 3x3 smoothing conv. "conv_refine" instead projects each patch to a
        feature map and blends patch seams with a small GroupNorm conv stack (sized by
        ``refine_channels``/``refine_kernel_size``, both unused for "linear"); this is
        the enhanced head used by the ``*_enhanced``/``*_residual`` baselines.
        """
        super().__init__(**kwargs)
        if decode_head not in ("linear", "conv_refine"):
            msg = f"Unsupported decode_head: {decode_head!r}."
            raise ValueError(msg)
        self.decode_head = decode_head

        # Ensure input is square and divisible by patch_size
        if self.data_space.shape[0] != self.data_space.shape[1]:
            msg = "The height and width of the input are not equal."
            raise ValueError(msg)
        if self.data_space.shape[0] % patch_size != 0:
            msg = f"img_size {self.data_space.shape[0]} must be divisible by patch_size {patch_size}."
            raise ValueError(msg)

        self.img_size = self.data_space.shape[0]
        self.patch_size = patch_size
        self.out_channels = self.data_space.channels
        self.refine_channels = refine_channels

        # The input is a window of n_history_steps timesteps concatenated along
        # channels (see BaseProcessor.rollout), so patch embedding must accept
        # n_history_steps times as many channels as a single timestep has.
        self.patch_embed = PatchEmbedding(
            self.data_space.channels * self.n_history_steps,
            patch_size,
            emb_dim,
            self.img_size,
        )
        num_patches = (self.img_size // patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, emb_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(emb_dim, heads, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(emb_dim)

        if decode_head == "linear":
            # Original head: per-patch linear decode straight to output values (each
            # token to patch_size * patch_size * out_channels), followed by a single
            # 3x3 smoothing conv.
            self.decoder = nn.Sequential(
                nn.Linear(emb_dim, patch_size * patch_size * self.out_channels),
            )
            self.smooth = nn.Conv2d(
                self.out_channels, self.out_channels, kernel_size=3, padding=1
            )
        else:
            # Enhanced head: project each patch token to a *feature* map at pixel
            # resolution, not directly to output values:
            # (B, N, patch_size * patch_size * refine_channels).
            self.patch_to_pixels = nn.Linear(
                emb_dim, patch_size * patch_size * refine_channels
            )

            # Each patch above is decoded independently, so the reassembled feature map
            # has hard seams at every patch boundary. Refine with a small conv stack to
            # blend across those seams before projecting down to the output channels --
            # a single 1-pixel-radius conv can't smooth a patch_size-wide discontinuity.
            # The stack's receptive field (4 * (refine_kernel_size - 1) + 1, for 2
            # blocks of 2 subblocks each) must exceed patch_size for it to see both
            # sides of a seam at once; the default kernel_size=3 only reaches 9px, so
            # seams wider than that (e.g. patch_size=12) are structurally unblendable
            # regardless of how long it trains. Uses GroupNorm rather than BatchNorm: it
            # normalises per-sample instead of off batch statistics, so it can't suffer
            # the train/eval divergence that collapsed the UNet processor at small batch
            # sizes.
            self.refine = nn.Sequential(
                CommonConvBlock(
                    refine_channels,
                    refine_channels,
                    kernel_size=refine_kernel_size,
                    n_subblocks=2,
                    norm_type="groupnorm",
                ),
                CommonConvBlock(
                    refine_channels,
                    refine_channels,
                    kernel_size=refine_kernel_size,
                    n_subblocks=2,
                    norm_type="groupnorm",
                ),
                nn.Conv2d(refine_channels, self.out_channels, kernel_size=1),
            )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward pass through the ViT model for a window of history/forecast timesteps.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total * n_history_steps, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        """
        batch, _, height, width = x.shape

        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)  # (B, N, D)

        # Project tokens back to pixels. The "linear" head goes straight to output
        # values (out_channels); the "conv_refine" head goes to a feature map
        # (refine_channels) that the refine stack then reduces to out_channels.
        if self.decode_head == "linear":
            x = self.decoder(x)  # (B, N, out_channels * patch_size * patch_size)
            channels = self.out_channels
        else:
            x = self.patch_to_pixels(x)  # (B, N, refine_channels * patch * patch)
            channels = self.refine_channels

        if height % self.patch_size or width % self.patch_size:
            msg = (
                f"Latent space height ({height}) and width ({width}) must each be "
                f"divisible by patch size ({self.patch_size})."
            )
            raise ValueError(msg)

        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        x = x.reshape(
            batch,
            h_patches,
            w_patches,
            channels,
            self.patch_size,
            self.patch_size,
        )
        # Shape is batch, channels, h_patches, patch_size, w_patches, patch_size
        x = x.permute(0, 3, 1, 4, 2, 5)

        # Shape is batch, channels, height, width
        x = x.reshape(batch, channels, self.img_size, self.img_size)

        return self.smooth(x) if self.decode_head == "linear" else self.refine(x)
