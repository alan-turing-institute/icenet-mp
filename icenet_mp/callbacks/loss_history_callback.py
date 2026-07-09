import json
import logging
from pathlib import Path

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback

logger = logging.getLogger(__name__)


class LossHistoryCallback(Callback):
    """Record train/validation loss at the end of every epoch to a local JSON file.

    Unlike W&B-backed logging, this does not depend on any external logger being
    attached to the trainer, so it also works in offline/CI runs.
    """

    def __init__(self, *, history_path: str) -> None:
        """Record per-epoch train/validation loss to ``history_path`` as JSON."""
        super().__init__()
        self.history_path = Path(history_path)
        self.train_loss: list[float] = []
        self.validation_loss: list[float] = []

    def _write(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(
            json.dumps(
                {"train_loss": self.train_loss, "validation_loss": self.validation_loss},
                indent=2,
            )
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        """Record the epoch's mean training loss."""
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get("train_loss")
        if metric is not None:
            self.train_loss.append(float(metric))
            self._write()

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:  # noqa: ARG002
        """Record the epoch's mean validation loss."""
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get("validation_loss")
        if metric is not None:
            self.validation_loss.append(float(metric))
            self._write()
