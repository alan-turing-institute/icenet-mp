"""One-shot (non-autoregressive) Vision Transformer processor.

Description:
    Predicts every forecast step in a single forward pass, instead of the default
    BaseProcessor.rollout behaviour of calling forward once per forecast step and
    feeding each prediction back in as history. This removes the
    n_forecast_steps-times cost of running the network repeatedly per training step,
    at the price of no longer conditioning each forecast day on the model's own
    previously predicted days (only on the original history window).
"""

from typing import Any

import torch
from torch import nn

from icenet_mp.models.common import (
    CommonConvBlock,
    PatchEmbedding,
    TransformerEncoderBlock,
)
from icenet_mp.types import ProcessorOutput, TensorNCHW, TensorNTCHW

from .base_processor import BaseProcessor


class VitDirectProcessor(BaseProcessor):
    """Vision Transformer processor that predicts all forecast steps in one pass."""

    def __init__(
        self,
        *,
        depth: int = 3,
        dropout: float = 0.3,
        emb_dim: int = 128,
        heads: int = 4,
        mlp_dim: int = 256,
        patch_size: int = 16,
        refine_channels: int = 64,
        refine_kernel_size: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize the one-shot Vision Transformer processor."""
        super().__init__(**kwargs)

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
        # channels (see rollout below), so patch embedding must accept
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

        # Project each patch token to a *feature* map at pixel resolution, not directly
        # to output values: (B, N, patch_size * patch_size * refine_channels). Shared
        # across all forecast steps -- only the final 1x1 conv below splits by step.
        self.patch_to_pixels = nn.Linear(
            emb_dim, patch_size * patch_size * refine_channels
        )

        # Same seam-blending rationale as VitProcessor (patches are decoded
        # independently, so the reassembled feature map has hard seams at every patch
        # boundary that a small conv stack must blend across). The final 1x1 conv
        # projects to out_channels * n_forecast_steps instead of just out_channels,
        # since this processor predicts every forecast day from one shared feature
        # map in a single pass: channel index = forecast_step * out_channels + channel
        # (see rollout's reshape, which must use the same ordering).
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
            nn.Conv2d(
                refine_channels,
                self.out_channels * self.n_forecast_steps,
                kernel_size=1,
            ),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Predict every forecast step in one pass from a window of history timesteps.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total * n_history_steps, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, n_latent_channels_total * n_forecast_steps, latent_height, latent_width),
            channels ordered as forecast_step * n_latent_channels_total + channel.

        """
        batch, _, height, width = x.shape

        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)  # (B, N, D)
        x = self.patch_to_pixels(x)  # (B, N, refine_channels * patch_size * patch_size)

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
            self.refine_channels,
            self.patch_size,
            self.patch_size,
        )
        # Shape is batch, refine_channels, h_patches, patch_size, w_patches, patch_size
        x = x.permute(0, 3, 1, 4, 2, 5)

        # Shape is batch, refine_channels, height, width
        x = x.reshape(batch, self.refine_channels, self.img_size, self.img_size)

        return self.refine(x)

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:  # noqa: ARG002
        """Predict all forecast steps in a single forward pass (no autoregression).

        Args:
            x: Encoded input TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
            y: Unused; accepted for interface compatibility with BaseProcessor.rollout.

        Returns:
            ProcessorOutput with prediction TensorNTCHW (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width).

        """
        batch, n_history, channels, height, width = x.shape
        # Equivalent to BaseProcessor.rollout's `cat(list(window), dim=1)`: history
        # steps ordered oldest to newest, each contributing `channels` channels.
        window_cat = x.reshape(batch, n_history * channels, height, width)
        output = self(window_cat)  # (batch, n_forecast_steps * channels, height, width)
        output = output.reshape(batch, self.n_forecast_steps, channels, height, width)
        return ProcessorOutput(prediction=output)
