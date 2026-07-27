"""Fast synthetic-data pipeline sanity check.

Trains and evaluates a real model+data configuration (e.g. ``baseline/synthetic_unet``)
against a small, deterministic moving-circle dataset instead of real sea-ice data, then
checks that the model actually learned. This is intended to catch pipeline bugs (shape
mismatches, broken history/forecast windowing, rollout regressions) and confirm a model
is learning at all, in seconds rather than the hours a real training run takes.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from icenet_mp.model_service import ModelService

from .report import plot_loss_curve

logger = logging.getLogger(__name__)

@dataclass
class SyntheticCheckResult:
    passed: bool
    reasons: list[str]
    train_loss: list[float]
    validation_loss: list[float]
    report_path: Path


def _load_loss_history(metrics_path: Path) -> dict[str, list[float]]:
    if not metrics_path.exists():
        msg = (
            f"Expected local metrics at {metrics_path}, but it does not exist. "
            "Check that the config includes LocalFileLogger."
        )
        raise FileNotFoundError(msg)

    history: dict[str, list[float]] = {"train_loss": [], "validation_loss": []}
    with metrics_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            for metric_name, losses in history.items():
                if metric_name in record:
                    losses.append(float(record[metric_name]))

    return history


def _clear_loss_history(metrics_path: Path) -> None:
    """Ensure a pipeline check only assesses metrics from its current run."""
    metrics_path.unlink(missing_ok=True)


def _check_learning(
    validation_loss: list[float], *, min_relative_improvement: float
) -> list[str]:
    reasons = []
    if len(validation_loss) < 2:  # noqa: PLR2004
        reasons.append("Not enough validation epochs recorded to assess learning.")
        return reasons

    # Compare against the best epoch reached, not the last: a short toy run can easily
    # overfit after finding a good minimum, and that's not a pipeline/learnability bug.
    first, best = validation_loss[0], min(validation_loss)
    if first <= 0:
        reasons.append(
            f"Initial validation loss is non-positive ({first}); cannot assess "
            f"relative improvement."
        )
        return reasons

    improvement = (first - best) / first
    if improvement < min_relative_improvement:
        reasons.append(
            f"Best validation loss only improved by {improvement:.1%} over the first "
            f"epoch (first={first:.4g}, best={best:.4g}); expected at least "
            f"{min_relative_improvement:.0%}."
        )
    return reasons


def run_synthetic_pipeline_check(
    config: DictConfig,
    *,
    output_dir: Path,
    max_epochs: int | None = None,
    min_relative_improvement: float = 0.3,
) -> SyntheticCheckResult:
    """Run a model+data config against synthetic data and check that it learns.

    Args:
        config: A composed Hydra config, e.g. from the ``synthetic_unet`` baseline.
        output_dir: Directory receiving checkpoints and the report.
        max_epochs: Optional override for ``train.trainer.max_epochs``.
        min_relative_improvement: Minimum fractional drop from the first epoch's
            validation loss to the best epoch's, required for the check to pass.

    Returns:
        A `SyntheticCheckResult` describing whether the check passed and why not.

    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "report" / "metrics.jsonl"
    _clear_loss_history(metrics_path)

    # Work on a detached copy so we never mutate the caller's config in place.
    config = DictConfig(OmegaConf.to_container(config, resolve=False))
    if max_epochs is not None:
        config["train"]["trainer"]["max_epochs"] = max_epochs

    service = ModelService.from_config(config)
    service.train()
    service.evaluate()

    history = _load_loss_history(metrics_path)
    train_loss = history.get("train_loss", [])
    validation_loss = history.get("validation_loss", [])

    plot_loss_curve(
        train_loss=train_loss,
        validation_loss=validation_loss,
        output_path=output_dir / "report" / "loss_curve.png",
    )

    reasons = _check_learning(
        validation_loss, min_relative_improvement=min_relative_improvement
    )
    passed = not reasons

    report_path = output_dir / "report" / "summary.json"
    report_path.write_text(
        json.dumps(
            {
                "passed": passed,
                "reasons": reasons,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            },
            indent=2,
        )
    )

    return SyntheticCheckResult(
        passed=passed,
        reasons=reasons,
        train_loss=train_loss,
        validation_loss=validation_loss,
        report_path=report_path,
    )
