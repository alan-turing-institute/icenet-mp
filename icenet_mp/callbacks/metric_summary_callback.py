import logging
import re
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

_FSS_NEIGHBOURHOOD_SIZE_RE = re.compile(r"^fss_(\d+)$")


def _metric_group(metric_name: str) -> str:
    """Group name for a metric, so parametrised variants share one plot.

    FSS is computed at several neighbourhood sizes (``fss_1``, ``fss_5``, ...); these
    are grouped under ``"fss"`` so they land on a single per-forecast-day plot instead
    of one plot each. All other metrics are their own group.
    """
    if metric_name.startswith("fss_"):
        return "fss"
    return metric_name


def _series_key(
    stage: str, metric_name: str, *, grouped: bool, multiple_stages: bool
) -> str:
    """Series key for a per-forecast-day plot: only vary on what actually varies."""
    if not grouped:
        return stage
    if multiple_stages:
        return f"{stage}_{metric_name}"
    return metric_name


class MetricSummaryCallback(Callback):
    """A callback to summarise metrics at the end of an epoch or a run."""

    def log_per_epoch_metrics(
        self, trainer: Trainer, metrics: MetricCollection, stage: str
    ) -> None:
        """Log per-epoch metrics to W&B."""
        # Skip logging during sanity checking to avoid logging incomplete metrics
        if trainer.sanity_checking:
            return

        # Compute the metric value (e.g., SIEError) across all batches
        for name, metric in metrics.items():
            if not metric._update_called:
                continue
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
        # Skip logging during sanity checking to avoid logging incomplete metrics
        if trainer.sanity_checking:
            return

        # Check that W&B is being used as a logger
        if not isinstance(run := get_wandb_run(trainer), wandb.Run):
            logger.warning(
                "W&B is not being used as a logger, cannot log per-run metrics!"
            )
            return

        # Extract the metric values (e.g., SIEError) across all batches
        # Only consider metrics that have a value for each forecast day
        values_per_forecast_day = self._collect_values_per_forecast_day(metrics)

        fss_metric_names = self._log_per_forecast_day_plots(
            run, values_per_forecast_day, multiple_stages=len(metrics) > 1
        )
        self._log_fss_vs_neighbourhood_size(
            run, values_per_forecast_day, fss_metric_names
        )

    def _collect_values_per_forecast_day(
        self, metrics: dict[str, MetricCollection]
    ) -> dict[str, dict[str, "Tensor"]]:
        """Collect metric values that have a value for each forecast day."""
        values_per_forecast_day: dict[str, dict[str, Tensor]] = defaultdict(dict)
        for stage, metric_collection in metrics.items():
            for metric_name, metric in metric_collection.items():
                if not metric._update_called:
                    continue
                metric_tensor: Tensor = metric.compute()
                if metric_tensor.reshape(-1).shape[0] > 1:
                    values_per_forecast_day[metric_name][stage] = metric_tensor
        return values_per_forecast_day

    def _log_per_forecast_day_plots(
        self,
        run: wandb.Run,
        values_per_forecast_day: dict[str, dict[str, "Tensor"]],
        *,
        multiple_stages: bool,
    ) -> list[str]:
        """Log a per-forecast-day plot for each metric group.

        Metrics that should share a single plot (e.g. FSS at several neighbourhood
        sizes) are grouped; all other metrics get one plot each. Returns the metric
        names in the "fss" group, for use in the FSS-vs-neighbourhood-size plot.
        """
        metric_names_by_group: dict[str, list[str]] = defaultdict(list)
        for metric_name in values_per_forecast_day:
            metric_names_by_group[_metric_group(metric_name)].append(metric_name)

        for group_name, metric_names in metric_names_by_group.items():
            grouped = len(metric_names) > 1
            series: dict[str, Tensor] = {
                _series_key(
                    stage, metric_name, grouped=grouped, multiple_stages=multiple_stages
                ): tensor
                for metric_name in metric_names
                for stage, tensor in values_per_forecast_day[metric_name].items()
            }

            keys = list(series.keys())
            days = list(range(1, len(series[keys[0]]) + 1))
            plot_name = f"{group_name}_per_forecast_day"
            run.log(
                {
                    plot_name: wandb.plot.line_series(
                        xs=days,
                        ys=[series[key].tolist() for key in keys],
                        keys=keys,
                        title=plot_name,
                        xname="day",
                    )
                },
            )

        return metric_names_by_group.get("fss", [])

    def _log_fss_vs_neighbourhood_size(
        self,
        run: wandb.Run,
        values_per_forecast_day: dict[str, dict[str, "Tensor"]],
        fss_metric_names: list[str],
    ) -> None:
        """Plot mean FSS (over forecast days) against neighbourhood size."""
        sizes_and_names = sorted(
            (int(match.group(1)), name)
            for name in fss_metric_names
            if (match := _FSS_NEIGHBOURHOOD_SIZE_RE.match(name))
        )
        if not sizes_and_names:
            return

        stages = list(values_per_forecast_day[sizes_and_names[0][1]])
        sizes = [size for size, _ in sizes_and_names]
        ys = [
            [
                values_per_forecast_day[name][stage].mean().item()
                for _, name in sizes_and_names
            ]
            for stage in stages
        ]

        # A flat reference line at FSS = 0.5, the usual "skilful" threshold
        keys = [*stages, "skilful_threshold"]
        ys = [*ys, [0.5] * len(sizes)]

        plot_name = "fss_vs_neighbourhood_size"
        run.log(
            {
                plot_name: wandb.plot.line_series(
                    xs=sizes,
                    ys=ys,
                    keys=keys,
                    title=plot_name,
                    xname="neighbourhood_size",
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
