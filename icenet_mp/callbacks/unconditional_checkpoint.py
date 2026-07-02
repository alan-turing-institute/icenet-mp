from pathlib import Path

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


class UnconditionalCheckpoint(Callback):
    """A callback to summarise metrics during evaluation."""

    def __init__(self, *, on_train_end: bool = False) -> None:
        """Save a checkpoint unconditionally.

        Args:
            on_train_end: Whether to save a checkpoint at the end of training

        """
        super().__init__()
        self.impl = ModelCheckpoint()
        self._on_train_end = on_train_end

    @property
    def dirpath(self) -> str | Path | None:
        """Return the directory path where checkpoints are saved."""
        return self.impl.dirpath

    @dirpath.setter
    def dirpath(self, value: str | Path | None) -> None:
        """Set the directory path where checkpoints are saved."""
        if value:
            self.impl.dirpath = Path(value)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        """Called when training ends."""
        if self._on_train_end:
            self.save_unconditionally(trainer)

    def save_unconditionally(self, trainer: Trainer) -> None:
        """Save a checkpoint unconditionally."""
        monitor_candidates = {
            "epoch": torch.tensor(trainer.current_epoch),
            "step": torch.tensor(trainer.global_step),
        }
        filepath = self.impl.format_checkpoint_name(monitor_candidates)
        trainer.save_checkpoint(filepath)
