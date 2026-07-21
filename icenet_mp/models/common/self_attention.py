import torch
from torch import Tensor, nn


class SelfAttention2D(nn.Module):
    """Multi-head, linear-complexity self-attention over NCHW tensors.

    Use a ReLU-kernel linear attention mechanism (Katharopoulos et al. 2020) as is done
    in the EfficientViT LiteMLA. This scales as O(N) in the spatial size (H*W) rather
    than O(N^2) in the standard softmax attention.
    """

    def __init__(self, channels: int, heads: int, eps: float = 1e-6) -> None:
        """Initialise a SelfAttention2D."""
        super().__init__()
        if channels % heads != 0:
            msg = f"channels ({channels}) must be divisible by heads ({heads})"
            raise ValueError(msg)
        self.heads = heads
        self.head_dim = channels // heads
        self.eps = eps
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU linear attention to x."""
        n, c, h, w = x.shape
        # [N, C, H, W] -> [N, 3*C, H, W] -> [N, 3, heads, head_dim, H*W]
        qkv = self.qkv(x).view(n, 3, self.heads, self.head_dim, h * w)
        q, k, v = qkv.unbind(dim=1)  # each is [N, heads, head_dim, H*W]
        q = q.relu()
        k = k.relu()

        # Linear attention: aggregate K^T V and the K normaliser once and apply to q
        # kv shape: [N, heads, head_dim, head_dim]
        kv = torch.einsum("bhdn,bhen->bhde", k, v)
        # k_norm shape: [N, heads, head_dim]
        k_norm = k.sum(dim=-1)
        # numerator shape: [N, heads, head_dim, H*W]
        numerator = torch.einsum("bhdn,bhde->bhen", q, kv)
        # denominator shape: [N, heads, 1, H*W]
        denominator = torch.einsum("bhdn,bhd->bhn", q, k_norm).unsqueeze(2) + self.eps

        # Reshape back to [N, C, H, W] and project to output channels
        return self.proj((numerator / denominator).reshape(n, c, h, w))
