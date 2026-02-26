import logging
from typing import TYPE_CHECKING

import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.loggers import WandbLogger

from icenet_mp.utils import get_wandb_run

if TYPE_CHECKING:
    from torchmetrics import MetricCollection


logger = logging.getLogger(__name__)


class MetricSummaryCallback(Callback):
    """A callback to summarise metrics during evaluation."""

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of testing."""
        test_metrics: MetricCollection = pl_module.test_metrics  # type: ignore[assignment]
        if not hasattr(test_metrics, "items"):
            logger.warning(
                "test_metrics does not have an items() method, skipping metric summary."
            )
            return

        for name, metric in test_metrics.items():
            # Compute the metric value (e.g., SIEError) across all batches and log it
            values = metric.compute()

            for logger_ in trainer.loggers:
                # check if WandB is being used as a logger, and if so, log the metric values as a table and plot
                if isinstance(logger_, WandbLogger):
                    # If the metric returns a value for each day, log it as a W&B table and plot it
                    if values.numel() > 1:
                        table = wandb.Table(
                            data=list(enumerate(values.tolist(), start=1)),
                            columns=["day", name],
                        )
                    plot_name = name + " per day"
                    get_wandb_run(trainer).log(
                        {
                            plot_name: wandb.plot.line(
                                table, "day", name, title=plot_name
                            )
                        }
                    )

                # Log the mean value of the metric across all days
                logger_.log_metrics({f"{name} (mean)": values.mean().item()})
