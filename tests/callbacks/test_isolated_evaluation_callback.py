import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from lightning import LightningModule, Trainer
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from icenet_mp.callbacks import IsolatedEvaluationCallback
from icenet_mp.metrics import MAEPerForecastDay
from icenet_mp.types import DataSpace, ModelStepOutput


def _configured_callback(output_dir: Path) -> IsolatedEvaluationCallback:
    callback = IsolatedEvaluationCallback(str(output_dir))
    checkpoint_path = output_dir.parent / "model.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    callback.set_evaluation_context(
        checkpoint_path=checkpoint_path,
        config=DictConfig({"predict": {"n_forecast_steps": 2}}),
        target_group_name="sic",
        target_variable_indices=[1],
    )
    return callback


def _module() -> MagicMock:
    module = MagicMock(spec=LightningModule)
    module.input_spaces = [DataSpace(channels=2, name="sic", shape=(1, 1))]
    module.test_metrics = MetricCollection({"mae": MAEPerForecastDay()})
    return module


def test_writes_per_lead_persistence_metrics_without_wandb(tmp_path: Path) -> None:
    """Persist all comparison values without consulting a logger or W&B."""
    callback = _configured_callback(tmp_path / "run-a")
    module = _module()
    trainer = MagicMock(spec=Trainer)
    trainer.is_global_zero = True
    callback.on_test_epoch_start(trainer, module)

    target = torch.tensor([1.0, 3.0]).view(1, 2, 1, 1, 1)
    prediction = torch.tensor([2.0, 1.0]).view(1, 2, 1, 1, 1)
    history = torch.tensor([9.0, 0.0, 8.0, 1.0]).view(1, 2, 2, 1, 1)
    module.test_metrics.update(prediction, target)
    callback.on_test_batch_end(
        trainer,
        module,
        ModelStepOutput(prediction, target, torch.tensor(0.0)),
        {"sic": history},
        0,
    )
    callback.on_test_epoch_end(trainer, module)

    result = json.loads((tmp_path / "run-a" / "evaluation_metrics.json").read_text())
    assert result["metrics"]["mae"]["model"] == [1.0, 2.0]
    assert result["metrics"]["mae"]["persistence"] == [0.0, 2.0]
    assert result["metrics"]["mae"]["absolute_delta"] == [1.0, 0.0]
    assert result["metrics"]["mae"]["skill"] == [None, 0.0]
    assert result["checkpoint_path"].endswith("model.ckpt")
    checkpoint_stat = (tmp_path / "model.ckpt").stat()
    assert result["checkpoint_identity"] == {
        "size_bytes": checkpoint_stat.st_size,
        "mtime_ns": checkpoint_stat.st_mtime_ns,
    }


def test_output_isolation_refuses_existing_result(tmp_path: Path) -> None:
    """Do not append to or replace an earlier named evaluation."""
    output_dir = tmp_path / "named-run"
    output_dir.mkdir()
    (output_dir / "evaluation_metrics.json").write_text("existing")

    with pytest.raises(FileExistsError, match="Select a new evaluation run name"):
        _configured_callback(output_dir).on_test_epoch_start(
            MagicMock(spec=Trainer), _module()
        )


def test_nonzero_rank_computes_all_metrics_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All ranks enter TorchMetrics compute reductions; only rank zero writes."""
    callback = _configured_callback(tmp_path / "distributed-run")
    module = _module()
    trainer = MagicMock(spec=Trainer)
    trainer.is_global_zero = False
    callback.on_test_epoch_start(trainer, module)
    assert callback.persistence_metrics is not None
    model_metric = module.test_metrics["mae"]
    persistence_metric = callback.persistence_metrics["mae"]
    model_compute = MagicMock(return_value=torch.tensor([1.0]))
    persistence_compute = MagicMock(return_value=torch.tensor([2.0]))
    monkeypatch.setattr(model_metric, "compute", model_compute)
    monkeypatch.setattr(persistence_metric, "compute", persistence_compute)

    callback.on_test_epoch_end(trainer, module)

    model_compute.assert_called_once()
    persistence_compute.assert_called_once()
    assert not callback.output_path.exists()


def test_fails_when_target_history_is_not_a_model_input(tmp_path: Path) -> None:
    """Report clearly when the configured persistence source is unavailable."""
    module = _module()
    module.input_spaces = [DataSpace(channels=1, name="weather", shape=(1, 1))]

    with pytest.raises(ValueError, match="requires target SIC history group 'sic'"):
        _configured_callback(tmp_path / "run").on_test_epoch_start(
            MagicMock(spec=Trainer), module
        )
