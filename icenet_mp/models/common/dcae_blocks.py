"""Building blocks for the Deep Compression AutoEncoder (DCAE) architecture.

Reference:
    Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
    (Chen et al., 2024)
"""

from typing import Any

import torch
from torch import Tensor, nn


class LayerNorm2D(nn.Module):
    """Standardise over the channel dimension of a ``(B, C, H, W)`` tensor."""

    def __init__(self, eps: float = 1e-5) -> None:
        """Initialise a LayerNorm2D."""
        super().__init__()
        self.register_buffer("eps", torch.as_tensor(eps))

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps)


class SelfAttention2D(nn.Module):
    """Multi-head self-attention over 2D spatial feature maps ``(B, C, H, W)``."""

    def __init__(self, channels: int, heads: int) -> None:
        """Initialise a SelfAttention2D."""
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=heads, batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention to x."""
        b, c, h, w = x.shape
        y = x.view(b, c, h * w).transpose(1, 2)  # (B, H*W, C)
        y, _ = self.mha(y, y, y, need_weights=False)
        return y.transpose(1, 2).view(b, c, h, w)


class ResBlock(nn.Module):
    """Residual block: normalise → (optional) attention → 1x1 FFN.

    Implementation of the DCAE ResBlock. The FFN always uses 1x1 convolutions regardless
    of the ``kernel_size`` passed via ``**conv_kwargs``.
    """

    def __init__(
        self,
        channels: int,
        *,
        norm: str = "group",
        groups: int = 16,
        attention_heads: int | None = None,
        ffn_factor: int = 1,
        dropout: float | None = None,
        **conv_kwargs: Any,
    ) -> None:
        """Initialise a ResBlock."""
        super().__init__()

        if norm == "layer":
            self.norm = LayerNorm2D()
        elif norm == "group":
            self.norm = nn.GroupNorm(
                num_groups=min(groups, channels), num_channels=channels, affine=False
            )
        else:
            msg = f"Unknown norm: {norm!r}"
            raise NotImplementedError(msg)

        self.attn = (
            SelfAttention2D(channels, heads=attention_heads)
            if attention_heads is not None
            else None
        )

        # FFN uses 1x1 convolutions regardless of outer kernel_size
        ffn_kwargs = dict(conv_kwargs)
        ffn_kwargs.update(kernel_size=1, padding=0)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_factor * channels, **ffn_kwargs),
            nn.SiLU(),
            *([] if dropout is None else [nn.Dropout(dropout)]),
            nn.Conv2d(ffn_factor * channels, channels, **ffn_kwargs),
        )
        self.ffn[-1].weight.data.mul_(1e-2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        if self.attn is not None:
            y = y + self.attn(y)
        return x + self.ffn(y)
