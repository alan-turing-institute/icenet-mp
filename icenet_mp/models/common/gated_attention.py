from torch import Tensor, nn


class GatedAttention(nn.Module):
    """Gated large-kernel attention: gSTA.

    - a depthwise conv (kernel 2*dilation-1)
    - a depthwise dilated conv (kernel ~= kernel_size // dilation)
    - a gated 1x1 projection.
    """

    def __init__(self, channels: int, *, kernel_size: int, dilation: int) -> None:
        super().__init__()
        depthwise_kernel = 2 * dilation - 1
        dilated_kernel = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)

        self.attend = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                depthwise_kernel,
                padding=(depthwise_kernel - 1) // 2,
                groups=channels,
            ),
            nn.Conv2d(
                channels,
                channels,
                dilated_kernel,
                padding=dilation * (dilated_kernel - 1) // 2,
                groups=channels,
                dilation=dilation,
            ),
        )
        self.expand = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.expand(self.attend(x)).chunk(2, dim=1)
        return x + self.proj_out(value * gate.sigmoid())
