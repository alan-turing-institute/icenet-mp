"""Learning-rate schedulers used by IceNet-MP."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineAnnealingLR(LRScheduler):
    """Linearly warm up, then cosine-decay without restarting after the horizon."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_epochs: int,
        warmup_epochs: int = 5,
        start_factor: float = 0.1,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        """Initialise the scheduler.

        Args:
            optimizer: Optimizer whose learning rate is scheduled.
            total_epochs: Training horizon over which warm-up and decay are applied.
            warmup_epochs: Number of initial epochs used for linear warm-up.
            start_factor: Fraction of each base learning rate used at the start.
            eta_min: Minimum learning rate after cosine decay.
            last_epoch: Last completed epoch, following PyTorch scheduler semantics.

        Raises:
            ValueError: If the scheduler configuration is invalid.

        """
        if total_epochs <= 0:
            msg = f"total_epochs must be positive, got {total_epochs}."
            raise ValueError(msg)
        if warmup_epochs < 0 or warmup_epochs >= total_epochs:
            msg = (
                "warmup_epochs must be non-negative and smaller than total_epochs, "
                f"got {warmup_epochs} and {total_epochs}."
            )
            raise ValueError(msg)
        if not 0.0 < start_factor <= 1.0:
            msg = f"start_factor must be in (0, 1], got {start_factor}."
            raise ValueError(msg)
        if eta_min < 0.0:
            msg = f"eta_min must be non-negative, got {eta_min}."
            raise ValueError(msg)

        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.start_factor = start_factor
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        """Return learning rates for the current scheduler epoch."""
        epoch = max(self.last_epoch, 0)

        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            factor = self.start_factor + (1.0 - self.start_factor) * progress
            return [base_lr * factor for base_lr in self.base_lrs]

        decay_epochs = self.total_epochs - self.warmup_epochs
        decay_progress = min(
            max((epoch - self.warmup_epochs) / decay_epochs, 0.0),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine
            for base_lr in self.base_lrs
        ]
