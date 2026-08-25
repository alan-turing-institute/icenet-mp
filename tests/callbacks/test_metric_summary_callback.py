import logging
from unittest.mock import MagicMock

import pytest
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer.states import TrainerFn
from torchmetrics import MeanAbsoluteError, MetricCollection

from icenet_mp.callbacks.metric_summary_callback import MetricSummaryCallback
from icenet_mp.metrics import (
    MAEPerForecastDay,
    RMSEPerForecastDay,
    SeaIceExtentErrorPerForecastDay,
)


@pytest.fixture
def callback() -> MetricSummaryCallback:
    """Create a MetricSummaryCallback instance."""
    return MetricSummaryCallback()


@pytest.fixture
def mock_trainer() -> MagicMock:
    """Create a mock Trainer."""
    trainer = MagicMock(spec=Trainer)
    trainer.sanity_checking = False
    mock_logger = MagicMock()
    trainer.loggers = [mock_logger]
    return trainer


@pytest.fixture
def mock_module() -> MagicMock:
    """Create a mock LightningModule."""
    return MagicMock(spec=LightningModule)


class TestOnTestEnd:
    """Tests for on_test_end method."""

    def test_on_test_end_with_metric_collection(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test on_test_end with a valid MetricCollection."""
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        mock_module.test_metrics = metric_collection

        # Create sample predictions and targets
        preds = torch.randn(10)
        targets = torch.randn(10)

        for pred, target in zip(preds, targets, strict=False):
            metric_collection.update(pred.unsqueeze(0), target.unsqueeze(0))

        callback.on_test_epoch_end(mock_trainer, mock_module)

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called()

    def test_on_test_end_with_invalid_test_metrics(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test on_test_end when test_metrics is not a MetricCollection."""
        mock_module.test_metrics = "invalid"

        callback.on_test_epoch_end(mock_trainer, mock_module)

        # Should not raise an error, just log a warning
        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_not_called()

    def test_on_test_end_with_wandb_logger_vector_metric(
        self,
        callback: MetricSummaryCallback,
        mock_module: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test on_test_end with WandbLogger and a metric returning a vector."""
        mock_wandb = MagicMock()
        mock_get_wandb_run = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.callbacks.metric_summary_callback.wandb", mock_wandb
        )
        monkeypatch.setattr(
            "icenet_mp.callbacks.metric_summary_callback.get_wandb_run",
            mock_get_wandb_run,
        )

        # Mock wandb.Run for isinstance check
        class MockWandbRun:
            def __init__(self) -> None:
                self.log = MagicMock()

        mock_wandb.Run = MockWandbRun

        # Create a trainer with WandbLogger
        trainer = MagicMock(spec=Trainer)
        trainer.sanity_checking = False
        wandb_logger = MagicMock(spec=WandbLogger)
        trainer.loggers = [wandb_logger]

        # Mock get_wandb_run to return a MockWandbRun instance
        mock_run = MockWandbRun()
        mock_get_wandb_run.return_value = mock_run

        # Create a metric that returns multiple values (daily metric)
        metric_collection = MetricCollection({"mae_daily": MAEPerForecastDay()})
        mock_module.test_metrics = metric_collection

        # Create sample 5D data: (batch=1, time=3, channels=1, height=2, width=2)
        preds = torch.randn(1, 3, 1, 2, 2)
        targets = torch.randn(1, 3, 1, 2, 2)
        metric_collection.update(preds, targets)

        mock_plot = MagicMock()
        mock_wandb.plot.line_series.return_value = mock_plot

        callback.teardown(trainer, mock_module, stage="test")

        # Assert that wandb.plot.line_series was called with the daily values
        mock_wandb.plot.line_series.assert_called_once()
        line_series_kwargs = mock_wandb.plot.line_series.call_args[1]
        assert line_series_kwargs["keys"] == ["test"]
        assert line_series_kwargs["title"] == "mae_daily_per_forecast_day"
        assert line_series_kwargs["xname"] == "day"

        # Assert that wandb.log was called with the plot under the correct key
        mock_run.log.assert_called_once()
        log_call_args = mock_run.log.call_args[0][0]
        assert "mae_daily_per_forecast_day" in log_call_args

    def test_on_test_end_without_wandb_logger_vector_metric(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test on_test_end with non-WandbLogger and a metric returning a vector."""
        # Create a metric that returns multiple values (daily metric)
        metric_collection = MetricCollection({"mae_daily": MAEPerForecastDay()})
        mock_module.test_metrics = metric_collection

        # Create sample 5D data: (batch=1, time=3, channels=1, height=2, width=2)
        preds = torch.randn(1, 3, 1, 2, 2)
        targets = torch.randn(1, 3, 1, 2, 2)
        metric_collection.update(preds, targets)

        callback.on_test_epoch_end(mock_trainer, mock_module)

        # Assert that the mean value was logged without wandb plotting
        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called_once()
        metrics_call_args = mock_logger.log_metrics.call_args[0][0]
        assert "test_mae_daily_mean" in metrics_call_args


class TestLogPerEpochMetrics:
    """Tests for log_per_epoch_metrics."""

    def test_skips_during_sanity_checking(
        self, callback: MetricSummaryCallback, mock_trainer: MagicMock
    ) -> None:
        """Do not log anything while Lightning's sanity check is running."""
        mock_trainer.sanity_checking = True
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))

        callback.log_per_epoch_metrics(mock_trainer, metric_collection, stage="test")

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_not_called()

    def test_skips_metrics_that_were_never_updated(
        self, callback: MetricSummaryCallback, mock_trainer: MagicMock
    ) -> None:
        """Skip metrics in the collection whose update() was never called."""
        metric_collection = MetricCollection(
            {"mae": MeanAbsoluteError(), "unused": MeanAbsoluteError()}
        )
        metric_collection["mae"].update(torch.zeros(1), torch.ones(1))

        callback.log_per_epoch_metrics(mock_trainer, metric_collection, stage="test")

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called_once()
        logged_metrics = mock_logger.log_metrics.call_args[0][0]
        assert "test_mae_mean" in logged_metrics
        assert "test_unused_mean" not in logged_metrics


class TestLogPerRunMetrics:
    """Tests for log_per_run_metrics."""

    def test_skips_during_sanity_checking(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Return before evaluating anything while Lightning's sanity check is running."""
        mock_trainer.sanity_checking = True

        with caplog.at_level(logging.WARNING):
            callback.log_per_run_metrics(mock_trainer, {})

        assert caplog.text == ""

    def test_warns_and_skips_without_wandb_logger(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn and skip logging when no WandbLogger/run is available."""
        with caplog.at_level(logging.WARNING):
            callback.log_per_run_metrics(mock_trainer, {})

        assert "W&B is not being used as a logger" in caplog.text

    def test_skips_metrics_that_were_never_updated(
        self,
        callback: MetricSummaryCallback,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Skip metrics whose update() was never called when building the per-day plot."""
        mock_wandb = MagicMock()
        mock_get_wandb_run = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.callbacks.metric_summary_callback.wandb", mock_wandb
        )
        monkeypatch.setattr(
            "icenet_mp.callbacks.metric_summary_callback.get_wandb_run",
            mock_get_wandb_run,
        )

        class MockWandbRun:
            def __init__(self) -> None:
                self.log = MagicMock()

        mock_wandb.Run = MockWandbRun
        mock_get_wandb_run.return_value = MockWandbRun()

        trainer = MagicMock(spec=Trainer)
        trainer.sanity_checking = False

        metric_collection = MetricCollection(
            {"mae_daily": MAEPerForecastDay(), "unused_daily": MAEPerForecastDay()}
        )
        preds = torch.randn(1, 3, 1, 2, 2)
        targets = torch.randn(1, 3, 1, 2, 2)
        metric_collection["mae_daily"].update(preds, targets)

        callback.log_per_run_metrics(trainer, {"test": metric_collection})

        mock_wandb.plot.line_series.assert_called_once()
        line_series_kwargs = mock_wandb.plot.line_series.call_args[1]
        assert line_series_kwargs["title"] == "mae_daily_per_forecast_day"


class TestEpochStartResets:
    """Tests for the on_*_epoch_start metric-reset hooks."""

    def test_on_test_epoch_start_resets_test_metrics(
        self, callback: MetricSummaryCallback, mock_trainer: MagicMock
    ) -> None:
        """Reset test_metrics at the start of a test epoch."""
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))
        pl_module = MagicMock(spec=LightningModule)
        pl_module.test_metrics = metric_collection

        callback.on_test_epoch_start(mock_trainer, pl_module)

        assert metric_collection["mae"]._update_called is False

    def test_on_train_epoch_start_resets_train_metrics(
        self, callback: MetricSummaryCallback, mock_trainer: MagicMock
    ) -> None:
        """Reset train_metrics at the start of a training epoch."""
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))
        pl_module = MagicMock(spec=LightningModule)
        pl_module.train_metrics = metric_collection

        callback.on_train_epoch_start(mock_trainer, pl_module)

        assert metric_collection["mae"]._update_called is False

    def test_on_validation_epoch_start_resets_validation_metrics(
        self, callback: MetricSummaryCallback, mock_trainer: MagicMock
    ) -> None:
        """Reset validation_metrics at the start of a validation epoch."""
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))
        pl_module = MagicMock(spec=LightningModule)
        pl_module.validation_metrics = metric_collection

        callback.on_validation_epoch_start(mock_trainer, pl_module)

        assert metric_collection["mae"]._update_called is False


class TestOnTrainEpochEnd:
    """Tests for on_train_epoch_end."""

    def test_logs_when_train_metrics_present(
        self, callback: MetricSummaryCallback, mock_trainer: MagicMock
    ) -> None:
        """Log per-epoch metrics when train_metrics is a MetricCollection."""
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))
        pl_module = MagicMock(spec=LightningModule)
        pl_module.train_metrics = metric_collection

        callback.on_train_epoch_end(mock_trainer, pl_module)

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called_once()

    def test_warns_when_train_metrics_missing(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn when train_metrics is not a MetricCollection."""
        pl_module = MagicMock(spec=LightningModule)
        pl_module.train_metrics = "invalid"

        with caplog.at_level(logging.WARNING):
            callback.on_train_epoch_end(mock_trainer, pl_module)

        assert "Could not load train metrics!" in caplog.text


class TestOnValidationEpochEnd:
    """Tests for on_validation_epoch_end."""

    def test_logs_when_validation_metrics_present(
        self, callback: MetricSummaryCallback, mock_trainer: MagicMock
    ) -> None:
        """Log per-epoch metrics when validation_metrics is a MetricCollection."""
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        metric_collection.update(torch.zeros(1), torch.ones(1))
        pl_module = MagicMock(spec=LightningModule)
        pl_module.validation_metrics = metric_collection

        callback.on_validation_epoch_end(mock_trainer, pl_module)

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called_once()

    def test_warns_when_validation_metrics_missing(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn when validation_metrics is not a MetricCollection."""
        pl_module = MagicMock(spec=LightningModule)
        pl_module.validation_metrics = "invalid"

        with caplog.at_level(logging.WARNING):
            callback.on_validation_epoch_end(mock_trainer, pl_module)

        assert "Could not load validation metrics!" in caplog.text


class TestTeardown:
    """Tests for teardown's per-stage metric collection."""

    def test_fitting_stage_collects_train_and_validation_metrics(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Gather both train and validation metrics for a fit run."""
        pl_module = MagicMock(spec=LightningModule)
        pl_module.train_metrics = MetricCollection({"mae": MeanAbsoluteError()})
        pl_module.validation_metrics = MetricCollection({"mae": MeanAbsoluteError()})
        mock_log_per_run_metrics = MagicMock()
        monkeypatch.setattr(callback, "log_per_run_metrics", mock_log_per_run_metrics)

        callback.teardown(mock_trainer, pl_module, stage=TrainerFn.FITTING.value)

        mock_log_per_run_metrics.assert_called_once()
        metrics = mock_log_per_run_metrics.call_args[0][1]
        assert set(metrics) == {"train", "validation"}

    def test_fitting_stage_warns_when_metrics_missing(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn for each stage whose metrics collection is missing during a fit run."""
        pl_module = MagicMock(spec=LightningModule)
        pl_module.train_metrics = "invalid"
        pl_module.validation_metrics = "invalid"

        with caplog.at_level(logging.WARNING):
            callback.teardown(mock_trainer, pl_module, stage=TrainerFn.FITTING.value)

        assert "Could not load train metrics!" in caplog.text
        assert "Could not load validation metrics!" in caplog.text

    def test_testing_stage_warns_when_test_metrics_missing(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn when test_metrics is missing during a test run."""
        pl_module = MagicMock(spec=LightningModule)
        pl_module.test_metrics = "invalid"

        with caplog.at_level(logging.WARNING):
            callback.teardown(mock_trainer, pl_module, stage=TrainerFn.TESTING.value)

        assert "Could not load test metrics!" in caplog.text


class TestMetricCalculations:
    """Tests for per-forecast-day metric calculation correctness."""

    def test_accumulates_multiple_batches(self) -> None:
        """Accumulate daily errors across batches with matching lead times."""
        metric = MAEPerForecastDay()
        metric.update(torch.zeros(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1, 1))
        metric.update(torch.zeros(1, 1, 1, 1, 1), torch.full((1, 1, 1, 1, 1), 3.0))

        assert torch.allclose(metric.compute(), torch.tensor([2.0]))

    def test_calculates_mean_mae_daily_correctly(self) -> None:
        """Test that MAE daily is calculated correctly."""
        # Convert 2D tensor to 5D tensor: (batch, channels, height, width, time)
        preds_2d = torch.tensor(
            [[1.0, 2.0, 4.0], [1.0, 3.0, 4.0], [2.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )
        targets_2d = torch.tensor(
            [[1.5, 2.5, 4.0], [0.5, 3.5, 4.0], [2.0, 4.0, 5.0], [2.5, 3.0, 6.0]]
        )

        # Reshape to 5D: (batch=1, time=3, channels=1, height=2, width=2)
        preds = preds_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        targets = targets_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        computed_mae = MAEPerForecastDay()
        computed_mae.update(preds, targets)
        daily_result = computed_mae.compute()
        # Expected MAE per day:
        # Day 1: (|1.0-1.5| + |1.0-0.5| + |2.0-2.0| + |2.0-2.5|) / 4 = 0.375
        # Day 2: (|2.0-2.5| + |3.0-3.5| + |3.0-4.0| + |4.0-3.0|) / 4 = 0.75
        # Day 3: (|4.0-4.0| + |4.0-4.0| + |5.0-5.0| + |6.0-6.0|) / 4 = 0.0
        expected_mae = torch.tensor([0.375, 0.75, 0.0])

        assert torch.allclose(daily_result, expected_mae, atol=1e-5)

        assert daily_result.mean().item() == pytest.approx(0.375, abs=1e-5)

    def test_calculates_mean_rmse_daily_correctly(self) -> None:
        """Test that RMSE daily is calculated correctly."""
        preds_2d = torch.tensor(
            [[1.0, 2.0, 4.0], [1.0, 3.0, 4.0], [2.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )
        targets_2d = torch.tensor(
            [[1.5, 2.5, 4.0], [0.5, 3.5, 4.0], [2.0, 4.0, 5.0], [2.5, 3.0, 6.0]]
        )

        # Reshape to 5D: (batch=1, time=3, channels=1, height=2, width=2)
        preds = preds_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        targets = targets_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        computed_rmse = RMSEPerForecastDay()
        computed_rmse.update(preds, targets)
        daily_result = computed_rmse.compute()

        # Expected RMSE per day:
        # Day 1: sqrt(mean([0.25, 0.25, 0.0, 0.25])) = sqrt(0.1875) = 0.4330127
        # Day 2: sqrt(mean([0.25, 0.25, 1.0, 1.0])) = sqrt(0.625) = 0.7905694
        # Day 3: sqrt(mean([0.0, 0.0, 0.0, 0.0])) = 0.0
        expected_rmse = torch.tensor([0.4330127, 0.7905694, 0.0])

        assert torch.allclose(daily_result, expected_rmse, atol=1e-5)

        assert daily_result.mean().item() == pytest.approx(0.40786, abs=1e-5)

    def test_calculates_mean_sieerror_daily_correctly(self) -> None:
        """Test that SIEError daily is calculated correctly."""
        preds_2d = torch.tensor(
            [[0.0, 0.1, 0.8], [0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.0, 0.1, 0.0]]
        )
        targets_2d = torch.tensor(
            [[0.3, 0.5, 0.1], [0.6, 0.1, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 1.0]]
        )

        # Reshape to 5D: (batch=1, time=3, channels=1, height=2, width=2)
        preds = preds_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        targets = targets_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        computed_sie = SeaIceExtentErrorPerForecastDay(pixel_size=1)
        computed_sie.update(preds, targets)
        daily_result = computed_sie.compute()

        # Expected SIEError per day:
        # Day 1: sie error = |0-1 + 0-1 + 1-1 + 0-0| * 1^2 = 2.0
        # Day 2: sie error = |0-1 + 1-0 + 1-1 + 0-0| * 1^2 = 0.0
        # Day 3: sie error = |1-0 + 1-0 + 1-1 + 0-1| * 1^2 = 1.0
        expected_sie = torch.tensor([2.0, 0.0, 1.0])  # pixel_size=1 -> no scaling

        assert torch.allclose(daily_result, expected_sie, atol=1e-5)

        assert daily_result.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_calculates_mean_sieerror_daily_pixel_size(self) -> None:
        """Test that SIEError daily is calculated correctly."""
        preds_2d = torch.tensor(
            [[0.0, 0.1, 0.8], [0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.0, 0.1, 0.0]]
        )
        targets_2d = torch.tensor(
            [[0.3, 0.5, 0.1], [0.6, 0.1, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 1.0]]
        )

        # Reshape to 5D: (batch=1, time=3, channels=1, height=2, width=2)
        preds = preds_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        targets = targets_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        computed_sie = SeaIceExtentErrorPerForecastDay()
        computed_sie.update(preds, targets)
        daily_result = computed_sie.compute()

        # Expected SIEError per day (before pixel-size scaling):
        # Day 1: sie error = |0-1 + 0-1 + 1-1 + 0-0| * 1^2 = 2.0
        # Day 2: sie error = |0-1 + 1-0 + 1-1 + 0-0| * 1^2 = 0.0
        # Day 3: sie error = |1-0 + 1-0 + 1-1 + 0-1| * 1^2 = 1.0
        expected_sie = torch.tensor(
            [1250.0, 0.0, 625.0]
        )  # default pixel_size=25 -> scaled by 25^2

        assert torch.allclose(daily_result, expected_sie, atol=1e-5)

        assert daily_result.mean().item() == pytest.approx(625.0, abs=1e-5)
