"""Building blocks for the Deep Compression AutoEncoder (DCAE) architecture.

Reference:
    Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
    (Chen et al., 2024)
"""

from typing import Any

from torch import Tensor, nn

from .normalisations import normalisation_from_name


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
