"""Residual block: normalise -> (optional) attention -> 1x1 feed-forward network."""

from typing import Any

from torch import Tensor, nn

from .normalisations import normalisation_from_name
from .self_attention import SelfAttention2D


class ResBlock(nn.Module):
    """Residual block: normalise -> (optional) attention -> spatial feed-forward network.

    Implementation of the DCAE ResBlock. The FFN uses the ``kernel_size`` (and other
    conv settings) passed via ``**conv_kwargs``, matching DCAE's use of spatial (not
    1x1) convolutions for local feature mixing.
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

        self.norm = normalisation_from_name(norm, channels)
        self.attn = (
            SelfAttention2D(channels, heads=attention_heads)
            if attention_heads is not None
            else None
        )

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
        y = self.norm(x)
        if self.attn is not None:
            y = y + self.attn(y)
        return x + self.ffn(y)
