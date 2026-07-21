"""Residual block: normalise -> (optional) attention -> 1x1 feed-forward network."""

from typing import Any

from torch import Tensor, nn

from .normalisations import normalisation_from_name
from .self_attention import SelfAttention2D


class ResBlock(nn.Module):
    """Residual block following the EfficientViTBlock design.

    - (optional) attention subblock
        - normalise the input
        - attention to select spatial features
        - add residual connection to the input
    - feed-forward subblock
        - normalise the input
        - 1x1 convolution to expand channels by `ffn_factor`
        - SiLU activation
        - (optional) dropout
        - 1x1 convolution to reduce channels back to input size
        - add residual connection to the input
    """

    def __init__(
        self,
        channels: int,
        *,
        norm: str = "groupnorm",
        attention_heads: int | None = None,
        ffn_factor: int = 1,
        dropout: float | None = None,
        **conv_kwargs: Any,
    ) -> None:
        """Initialise a ResBlock."""
        super().__init__()

        # Optional attention sub-block
        self.attn_norm: nn.Module | None = (
            normalisation_from_name(norm, channels)
            if attention_heads is not None
            else None
        )
        self.attn: nn.Module | None = (
            SelfAttention2D(channels, heads=attention_heads)
            if attention_heads is not None
            else None
        )

        # Feed-forward sub-block
        self.ffn_norm = normalisation_from_name(norm, channels)
        ffn_layers: list[nn.Module] = [
            nn.Conv2d(channels, ffn_factor * channels, **conv_kwargs),
            nn.SiLU(),
        ]
        if dropout is not None:
            ffn_layers.append(nn.Dropout(dropout))
        conv_layer = nn.Conv2d(ffn_factor * channels, channels, **conv_kwargs)
        conv_layer.weight.data.mul_(1e-2)
        ffn_layers.append(conv_layer)
        self.ffn = nn.Sequential(*ffn_layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.attn is not None and self.attn_norm is not None:
            x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))
