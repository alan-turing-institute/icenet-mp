from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.metrics import (
    IntegratedIceEdgeErrorPerForecastDay,
    MAEPerForecastDay,
    RMSEPerForecastDay,
    SeaIceExtentErrorPerForecastDay,
)
from icenet_mp.models import BaseModel
from icenet_mp.types import TensorNTCHW


class ForecastEcho(BaseModel):
    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Return fixed predictions to isolate metric accumulation."""
        return inputs["forecast"]


def _score_batches(
    stage: str,
    metric_names: list[str],
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    mask_dir: Path,
) -> dict[str, torch.Tensor]:
    """Compute scores through the real model step for the selected stage."""
    model = ForecastEcho(
        hemisphere="north",
        input_spaces=[DictConfig({"name": "forecast", "channels": 1, "shape": [1, 1]})],
        output_space=DictConfig({"name": "target", "channels": 1, "shape": [1, 1]}),
        mask_dir=mask_dir,
        n_history_steps=2,
        n_forecast_steps=2,
        name="metric-batch-regression",
        metrics=metric_names,
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
        optimizer=DictConfig({}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
    )
    step, metrics = {
        "train": (model.training_step, model.train_metrics),
        "validation": (model.validation_step, model.validation_metrics),
        "test": (model.test_step, model.test_metrics),
    }[stage]
    with patch.object(model, "log"):
        for batch_idx, start in enumerate(range(0, prediction.shape[0], batch_size)):
            step(
                {
                    "forecast": prediction[start : start + batch_size],
                    "target": target[start : start + batch_size],
                },
                batch_idx,
            )
    return metrics.compute()


@pytest.mark.parametrize("stage", ["train", "validation", "test"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_signed_extent_error_is_independent_of_batch_size(
    stage: str, batch_size: int, tmp_path: Path
) -> None:
    """Equal overprediction and underprediction cancel only the signed error."""
    prediction = torch.tensor([1.0, 0.0]).reshape(2, 1, 1, 1, 1).repeat(1, 2, 1, 1, 1)
    target = 1.0 - prediction
    scores = _score_batches(
        stage, ["iiee", "sieerror"], prediction, target, batch_size, tmp_path
    )

    # One 25 km by 25 km ocean cell disagrees in each sample and forecast lead.
    for name, metric, expected in (
        ("iiee", IntegratedIceEdgeErrorPerForecastDay(), torch.full((2,), 625.0)),
        ("sieerror", SeaIceExtentErrorPerForecastDay(), torch.zeros(2)),
    ):
        metric.update(prediction, target)
        torch.testing.assert_close(metric.compute(), expected)
        torch.testing.assert_close(scores[name], expected)


@pytest.mark.parametrize("stage", ["train", "validation", "test"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_rmse_is_independent_of_batch_size(
    stage: str, batch_size: int, tmp_path: Path
) -> None:
    """MAE and RMSE must retain their different update rules across batches."""
    prediction = torch.tensor([1.0, 0.5]).reshape(2, 1, 1, 1, 1).repeat(1, 2, 1, 1, 1)
    target = torch.zeros_like(prediction)
    scores = _score_batches(
        stage, ["mae", "rmse"], prediction, target, batch_size, tmp_path
    )

    for name, metric, expected in (
        ("mae", MAEPerForecastDay(), torch.full((2,), 0.75)),
        ("rmse", RMSEPerForecastDay(), torch.full((2,), 0.625).sqrt()),
    ):
        metric.update(prediction, target)
        torch.testing.assert_close(metric.compute(), expected)
        torch.testing.assert_close(scores[name], expected)
