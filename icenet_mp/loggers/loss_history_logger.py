"""Lightning logger that records per-epoch train/validation loss to JSON.

Replaces ``LossHistoryCallback`` with a proper ``Logger`` subclass so it integrates
with the same infrastructure as ``LocalFileLogger`` and ``WandbLogger``, rather than
hooking into trainer lifecycle events via callbacks.
"""

import json
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers.logger import Logger


class LossHistoryLogger(Logger):
    """Record per-epoch train/validation loss to a local JSON file.

    Note: this is intentionally separate from ``LocalFileLogger`` because the two serve
    different purposes — ``LocalFileLogger`` streams every metric to JSONL in real time,
    while this logger accumulates only ``train_loss`` / ``validation_loss`` and writes a
    compact summary at the end of training.  If more batch-summary loggers are needed in
    future they can share a common base class.
    """

    def __init__(self, history_path: str) -> None:
        """Write per-epoch losses to ``history_path`` as JSON.

        Args:
            history_path: File path for the output JSON (e.g. ``report/loss_history.json``).
        """
        super().__init__()
        self._history_path = Path(history_path)
        self._train_loss: list[float] = []
        self._validation_loss: list[float] = []

    @property
    def name(self) -> str:
        """Return the logger name."""
        return "loss_history"

    @property
    def version(self) -> str:
        """Return a fixed version."""
        return "0"

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Do nothing; hyperparameters are saved as ``model_config.yaml``."""
        del params, args, kwargs

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:  # type: ignore[override]
        """Accumulate train/validation loss from Lightning's epoch-end metric calls.

        Lightning calls ``log_metrics()`` after each epoch when ``on_epoch=True`` is set
        on the original ``self.log()`` call in training/validation steps.
        """
        if "train_loss" in metrics:
            self._train_loss.append(float(metrics["train_loss"]))
        if "validation_loss" in metrics:
            self._validation_loss.append(float(metrics["validation_loss"]))

    def after_fit(self) -> None:
        """Write accumulated losses to JSON when training finishes."""
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history_path.write_text(
            json.dumps(
                {
                    "train_loss": self._train_loss,
                    "validation_loss": self._validation_loss,
                },
                indent=2,
            )
        )
