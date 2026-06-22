from torch import Tensor, nn


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
