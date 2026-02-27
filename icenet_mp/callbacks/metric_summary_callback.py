import logging

import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torchmetrics import MetricCollection

from icenet_mp.utils import get_wandb_run

logger = logging.getLogger(__name__)


class MetricSummaryCallback(Callback):
    """A callback to summarise metrics during evaluation."""

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of testing."""
        test_metrics: MetricCollection = pl_module.test_metrics  # type: ignore[assignment]
        if not isinstance(pl_module.test_metrics, MetricCollection):
            logger.warning("Could not load test metrics!")
            return

        for name, metric in test_metrics.items():
            # Compute the metric value (e.g., SIEError) across all batches and log it
            values = metric.compute()

            for logger_ in trainer.loggers:
                # Log the mean value of the metric across all days
                logger_.log_metrics({f"{name} (mean)": values.mean().item()})

            # check if WandB is being used as a logger and metrics are calculated for multiple days
            # if so, log the metric values as a table and plot
            if (
                isinstance(run := get_wandb_run(trainer), wandb.Run)
                and values.numel() > 1
            ):
                table = wandb.Table(
                    data=list(enumerate(values.tolist(), start=1)),
                    columns=["day", name],
                )
                plot_name = name + " per day"
                run.log(
                    {plot_name: wandb.plot.line(table, "day", name, title=plot_name)}
                )
