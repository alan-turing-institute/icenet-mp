import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MetricCollection

from icenet_mp.types import ModelStepOutput, TensorNTCHW


class IsolatedEvaluationCallback(Callback):
    """Write model and persistence evaluation metrics to one local artefact."""

    def __init__(self, output_dir: str) -> None:
        """Set the explicitly selected output directory."""
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_path = self.output_dir / "evaluation_metrics.json"
        self.persistence_metrics: MetricCollection | None = None
        self.checkpoint_path: Path | None = None
        self.checkpoint_identity: dict[str, int] | None = None
        self.target_group_name: str | None = None
        self.target_variable_indices: list[int] | None = None
        self.resolved_config: dict[str, Any] | None = None

    def set_evaluation_context(
        self,
        *,
        checkpoint_path: Path,
        config: DictConfig,
        target_group_name: str,
        target_variable_indices: list[int],
    ) -> None:
        """Supply checkpoint and resolved data provenance from ``ModelService``."""
        self.checkpoint_path = checkpoint_path.resolve()
        checkpoint_stat = self.checkpoint_path.stat()
        self.checkpoint_identity = {
            "size_bytes": checkpoint_stat.st_size,
            "mtime_ns": checkpoint_stat.st_mtime_ns,
        }
        self.target_group_name = target_group_name
        self.target_variable_indices = list(target_variable_indices)
        self.resolved_config = OmegaConf.to_container(
            config, resolve=True, enum_to_str=True
        )  # type: ignore[assignment]

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Validate persistence inputs and initialise like-for-like metrics."""
        del trainer
        if self.checkpoint_path is None or self.resolved_config is None:
            msg = "Isolated evaluation callback has no checkpoint evaluation context."
            raise RuntimeError(msg)
        if self.output_path.exists():
            msg = (
                f"Isolated evaluation output already exists: {self.output_path}. "
                "Select a new evaluation run name or remove the existing artefact."
            )
            raise FileExistsError(msg)
        test_metrics = getattr(pl_module, "test_metrics", None)
        if not isinstance(test_metrics, MetricCollection):
            msg = "Persistence evaluation requires model test_metrics."
            raise TypeError(msg)
        input_names = {space.name for space in getattr(pl_module, "input_spaces", [])}
        if self.target_group_name not in input_names:
            msg = (
                f"Persistence requires target SIC history group "
                f"{self.target_group_name!r}, but model inputs are {sorted(input_names)}."
            )
            raise ValueError(msg)
        if not self.target_variable_indices:
            msg = "Persistence requires at least one historical target SIC channel."
            raise ValueError(msg)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        test_metrics.reset()
        self.persistence_metrics = deepcopy(test_metrics)
        self.persistence_metrics.reset()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: dict[str, TensorNTCHW],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update persistence metrics from the latest target history frame."""
        del trainer, pl_module, batch_idx, dataloader_idx
        if self.persistence_metrics is None:
            msg = "Persistence metrics were not initialised."
            raise RuntimeError(msg)
        if self.target_group_name not in batch:
            msg = (
                f"Persistence requires target SIC history group "
                f"{self.target_group_name!r} in every evaluation batch."
            )
            raise ValueError(msg)
        if isinstance(outputs, ModelStepOutput):
            step_output = outputs
        elif isinstance(outputs, Mapping):
            step_output = ModelStepOutput(
                prediction=outputs["prediction"],
                target=outputs["target"],
                loss=outputs["loss"],
            )
        else:
            msg = "Persistence evaluation requires model step prediction and target outputs."
            raise TypeError(msg)
        history = batch[self.target_group_name]
        indices = self.target_variable_indices or []
        if history.ndim != len("NTCHW") or max(indices) >= history.shape[2]:
            msg = (
                "Persistence target SIC channel is unavailable in history tensor with "
                f"shape {tuple(history.shape)}; requested channel indices {indices}."
            )
            raise ValueError(msg)
        latest = history[:, -1:, indices]
        persistence = latest.expand(-1, step_output.target.shape[1], -1, -1, -1)
        if persistence.shape != step_output.target.shape:
            msg = (
                f"Persistence shape {tuple(persistence.shape)} does not match target "
                f"shape {tuple(step_output.target.shape)}."
            )
            raise ValueError(msg)
        self.persistence_metrics.update(persistence, step_output.target)

    @staticmethod
    def _values(metric: torch.Tensor) -> list[float]:
        return [float(value) for value in metric.detach().cpu().reshape(-1)]

    @staticmethod
    def _skill(
        name: str, model: torch.Tensor, persistence: torch.Tensor
    ) -> list[float | None]:
        if name == "accuracy":
            denominator = 100.0 - persistence
            values = (model - persistence) / denominator
        elif name in {"mae", "rmse", "sieerror", "centroid_error"}:
            denominator = persistence
            values = 1.0 - model / denominator
        else:
            return [None] * model.numel()
        return [
            None if zero else float(value)
            for value, zero in zip(
                values.detach().cpu().reshape(-1),
                denominator.detach().cpu().reshape(-1) == 0,
                strict=True,
            )
        ]

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Atomically write per-lead model, persistence, delta, and skill values."""
        if self.persistence_metrics is None:
            msg = "Persistence metrics were not initialised."
            raise RuntimeError(msg)
        test_metrics = getattr(pl_module, "test_metrics", None)
        if not isinstance(test_metrics, MetricCollection):
            msg = "Persistence evaluation requires model test_metrics."
            raise TypeError(msg)
        metrics: dict[str, Any] = {}
        for name, model_metric in test_metrics.items():
            persistence_metric = self.persistence_metrics[name]
            model = model_metric.compute().detach()
            persistence = persistence_metric.compute().to(model.device).detach()
            metrics[name] = {
                "model": self._values(model),
                "persistence": self._values(persistence),
                "absolute_delta": self._values(model - persistence),
                "skill": self._skill(name, model, persistence),
            }
        payload = {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_identity": self.checkpoint_identity,
            "target_history": {
                "group_name": self.target_group_name,
                "variable_indices": self.target_variable_indices,
                "source_timestep": "latest",
            },
            "definitions": {
                "absolute_delta": "model minus persistence in metric units",
                "error_skill": "1 - model / persistence",
                "accuracy_skill": "(model - persistence) / (100 - persistence)",
            },
            "metrics": metrics,
            "resolved_config": self.resolved_config,
        }
        # TorchMetrics synchronises distributed state during compute(), so every rank
        # must reach the computations above even though only rank zero writes.
        if not trainer.is_global_zero:
            return
        temporary_path = self.output_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(self.output_path)
