import torch
from torch import nn


class NormalisedFold(nn.Module):
    """Fold patches into an image while accounting for per-pixel overlaps."""

    def __init__(
        self,
        *,
        output_size: tuple[int, int],
        kernel_size: tuple[int, int],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
    ) -> None:
        """Initialise a NormalisedFold."""
        super().__init__()
        self.fold = nn.Fold(
            output_size=output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Compute an overlap mask by folding and unfolding a tensor of ones shaped like
        # the expected output. We disable gradients and mark the mask as a buffer to
        # ensure it is not treated as a learnable parameter.
        with torch.no_grad():
            ones = torch.ones(1, 1, *output_size)
            unfold = nn.Unfold(
                kernel_size=self.fold.kernel_size,
                stride=self.fold.stride,
                padding=self.fold.padding,
            )
            self.register_buffer(
                "overlap_mask", self.fold(unfold(ones)), persistent=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fold and normalize by overlap mask."""
        return self.fold(x) / self.overlap_mask
