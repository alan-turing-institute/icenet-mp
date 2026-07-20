import torch
from torch import nn


class DiceHuberLoss(nn.Module):
    """HuberLoss plus a soft-Dice term.

    A thin boundary ring is a tiny minority of pixels against a large interior and
    background, so a plain per-pixel loss gives it weak gradient pressure and the
    boundary is the slowest thing to sharpen. Dice is computed directly on the
    continuous prediction/target values (a 'soft' Dice) rather than a hard threshold,
    so it stays differentiable everywhere, and it does not scale with pixel count the
    way HuberLoss does, so it weights shape overlap regardless of how small the
    positive region is.
    """

    def __init__(
        self, delta: float = 0.5, dice_weight: float = 1.0, eps: float = 1e-6
    ) -> None:
        """Initialise DiceHuberLoss.

        Args:
            delta: HuberLoss's quadratic-to-linear transition threshold.
            dice_weight: Weight applied to the soft-Dice term before adding to Huber.
            eps: Smoothing term keeping the Dice ratio finite when both prediction and
                target are all-zero.

        """
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute HuberLoss(prediction, target) + dice_weight * soft_dice_loss."""
        huber = self.huber(prediction, target)
        intersection = (prediction * target).sum()
        union = prediction.sum() + target.sum()
        soft_dice_loss = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)
        return huber + self.dice_weight * soft_dice_loss
