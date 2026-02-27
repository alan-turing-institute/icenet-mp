from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanAbsoluteError, MetricCollection

from icenet_mp.callbacks.metric_summary_callback import MetricSummaryCallback
from icenet_mp.metrics.base_metrics import MAEPerForecastDay, RMSEPerForecastDay
from icenet_mp.metrics.sie_error_abs import SeaIceExtentErrorPerForecastDay


@pytest.fixture
def callback() -> MetricSummaryCallback:
    """Create a MetricSummaryCallback instance."""
    return MetricSummaryCallback()


@pytest.fixture
def mock_trainer() -> MagicMock:
    """Create a mock Trainer."""
    trainer = MagicMock(spec=Trainer)
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

        callback.on_test_end(mock_trainer, mock_module)

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

        callback.on_test_end(mock_trainer, mock_module)

        # Should not raise an error, just log a warning
        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_not_called()

    @patch("icenet_mp.callbacks.metric_summary_callback.get_wandb_run")
    @patch("icenet_mp.callbacks.metric_summary_callback.wandb")
    def test_on_test_end_with_wandb_logger_vector_metric(
        self,
        mock_wandb: MagicMock,
        mock_get_wandb_run: MagicMock,
        callback: MetricSummaryCallback,
        mock_module: MagicMock,
    ) -> None:
        """Test on_test_end with WandbLogger and a metric returning a vector."""

        # Mock wandb.Run for isinstance check
        class MockWandbRun:
            def __init__(self) -> None:
                self.log = MagicMock()

        mock_wandb.Run = MockWandbRun

        # Create a trainer with WandbLogger
        trainer = MagicMock(spec=Trainer)
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

        # Mock wandb.Table and wandb.plot.line
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table
        mock_plot = MagicMock()
        mock_wandb.plot.line.return_value = mock_plot

        callback.on_test_end(trainer, mock_module)

        # Assert that wandb.Table was created with the daily values
        mock_wandb.Table.assert_called_once()
        table_call_args = mock_wandb.Table.call_args
        assert table_call_args[1]["columns"] == ["day", "mae_daily"]
        # Verify data includes enumeration starting from 1
        data = table_call_args[1]["data"]
        assert len(data) == 3  # 3 days
        assert data[0][0] == 1  # First day index

        # Assert that wandb.plot.line was called
        mock_wandb.plot.line.assert_called_once_with(
            mock_table, "day", "mae_daily", title="mae_daily per day"
        )

        # Assert that wandb.log was called with the plot
        mock_run.log.assert_called_once()
        log_call_args = mock_run.log.call_args[0][0]
        assert "mae_daily per day" in log_call_args

        # Assert that the mean value was logged
        wandb_logger.log_metrics.assert_called_once()
        metrics_call_args = wandb_logger.log_metrics.call_args[0][0]
        assert "mae_daily (mean)" in metrics_call_args

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

        callback.on_test_end(mock_trainer, mock_module)

        # Assert that the mean value was logged without wandb plotting
        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called_once()
        metrics_call_args = mock_logger.log_metrics.call_args[0][0]
        assert "mae_daily (mean)" in metrics_call_args


class TestMetricCalculations:
    """Tests for on_test_end method."""

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
