import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.trainer.states import TrainerFn
from torchmetrics import MetricCollection

from icenet_mp.utils import get_wandb_run

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class MetricSummaryCallback(Callback):
    """A callback to summarise metrics at the end of an epoch or a run."""

    def log_per_epoch_metrics(
        self, trainer: Trainer, metrics: MetricCollection, stage: str
    ) -> None:
        """Log per-epoch metrics to W&B."""
        for name, metric in metrics.items():
            # Compute the metric value (e.g., SIEError) across all batches
            values: Tensor = metric.compute()

            # Log the mean value of the metric across all days
            for logger_ in trainer.loggers:
                logger_.log_metrics(
                    {
                        f"{stage}_{name}_mean".lower(): values.mean().item(),
                        "epoch": trainer.current_epoch,
                    }
                )

    def log_per_run_metrics(
        self, trainer: Trainer, metrics: dict[str, MetricCollection]
    ) -> None:
        """Log per-run metrics to W&B.

        Note that these will be based on metrics accumulated during the final epoch, due
        to the reset behaviour in log_per_epoch_metrics.
        """
        # Check that W&B is being used as a logger
        if not isinstance(run := get_wandb_run(trainer), wandb.Run):
            logger.warning(
                "W&B is not being used as a logger, cannot log per-run metrics!"
            )
            return

        # Extract the metric values (e.g., SIEError) across all batches
        # Only consider metrics that have a value for each forecast day
        values_per_forecast_day: dict[str, dict[str, Tensor]] = defaultdict(dict)
        for stage, metric_collection in metrics.items():
            for metric_name, metric in metric_collection.items():
                metric_tensor: Tensor = metric.compute()
                if metric_tensor.reshape(-1).shape[0] > 1:
                    values_per_forecast_day[metric_name][stage] = metric_tensor

        # For each metric, log the per-day values to W&B
        for metric_name, per_stage_metrics in values_per_forecast_day.items():
            stages = list(per_stage_metrics.keys())
            days = list(range(1, len(per_stage_metrics[stages[0]]) + 1))
            plot_name = f"{metric_name}_per_forecast_day"
            run.log(
                {
                    plot_name: wandb.plot.line_series(
                        xs=days,
                        ys=[per_stage_metrics[stage].tolist() for stage in stages],
                        keys=stages,
                        title=plot_name,
                        xname="day",
                    )
                },
            )

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        """Called at the start of a test epoch."""
        if isinstance(pl_module.test_metrics, MetricCollection):
            pl_module.test_metrics.reset()

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of a test epoch."""
        if isinstance(pl_module.test_metrics, MetricCollection):
            self.log_per_epoch_metrics(trainer, pl_module.test_metrics, stage="test")
        else:
            logger.warning("Could not load test metrics!")

    def on_train_epoch_start(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,
    ) -> None:
        """Called at the start of a train epoch."""
        if isinstance(pl_module.train_metrics, MetricCollection):
            pl_module.train_metrics.reset()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of a training epoch."""
        if isinstance(pl_module.train_metrics, MetricCollection):
            self.log_per_epoch_metrics(trainer, pl_module.train_metrics, stage="train")
        else:
            logger.warning("Could not load train metrics!")

    def on_validation_epoch_start(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,
    ) -> None:
        """Called at the start of a validation epoch."""
        if isinstance(pl_module.validation_metrics, MetricCollection):
            pl_module.validation_metrics.reset()

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called at the end of a validation epoch."""
        if isinstance(pl_module.validation_metrics, MetricCollection):
            self.log_per_epoch_metrics(
                trainer, pl_module.validation_metrics, stage="validation"
            )
        else:
            logger.warning("Could not load validation metrics!")

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Called at the end of a run."""
        metrics = {}
        # If this was a training run we want to log train and validation metrics
        if stage == TrainerFn.FITTING.value:
            if isinstance(pl_module.train_metrics, MetricCollection):
                metrics["train"] = pl_module.train_metrics
            else:
                logger.warning("Could not load train metrics!")
            if isinstance(pl_module.validation_metrics, MetricCollection):
                metrics["validation"] = pl_module.validation_metrics
            else:
                logger.warning("Could not load validation metrics!")
        # If this was a testing run we want to log test metrics
        elif stage == TrainerFn.TESTING.value:
            if isinstance(pl_module.test_metrics, MetricCollection):
                metrics["test"] = pl_module.test_metrics
            else:
                logger.warning("Could not load test metrics!")
        # Log the metrics
        self.log_per_run_metrics(trainer, metrics)
