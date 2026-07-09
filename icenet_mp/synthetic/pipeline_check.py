"""Fast synthetic-data pipeline sanity check.

Trains and evaluates a real model+data configuration (e.g. ``baseline/synthetic_unet``)
against a small, deterministic moving-circle dataset instead of real sea-ice data, then
checks that validation loss actually improved. This is intended to catch pipeline bugs
(shape mismatches, broken history/forecast windowing, rollout regressions) and confirm a
model is learning at all, in seconds rather than the hours a real training run takes.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from icenet_mp.model_service import ModelService

from .report import plot_loss_curve
from .shapes import MovingCircleConfig, generate_moving_circle_frames
from .zarr_writer import write_synthetic_zarr

logger = logging.getLogger(__name__)

SYNTHETIC_DATASET_NAME = "synthetic_sic"


@dataclass
class SyntheticCheckResult:
    passed: bool
    reasons: list[str]
    train_loss: list[float]
    validation_loss: list[float]
    report_path: Path


def _generate_dataset(config: DictConfig, output_dir: Path) -> None:
    dataset_entries = list(config["data"]["datasets"].values())
    unknown = [
        entry["name"] for entry in dataset_entries if entry["name"] != SYNTHETIC_DATASET_NAME
    ]
    if unknown:
        msg = (
            f"Don't know how to generate synthetic data for dataset(s) {unknown}. "
            f"The synthetic pipeline check currently only supports a single dataset "
            f"named '{SYNTHETIC_DATASET_NAME}' -- use the 'data=synthetic' config group."
        )
        raise ValueError(msg)

    frames = generate_moving_circle_frames(MovingCircleConfig())
    zarr_path = output_dir / "data" / "anemoi" / f"{SYNTHETIC_DATASET_NAME}.zarr"
    write_synthetic_zarr(zarr_path, frames=frames)
    logger.info("Wrote synthetic moving-circle dataset to %s.", zarr_path)


def _load_loss_history(history_path: Path) -> dict[str, list[float]]:
    if not history_path.exists():
        msg = (
            f"Expected loss history at {history_path}, but it does not exist. Does the "
            f"config include the 'loss_history' train callback?"
        )
        raise FileNotFoundError(msg)
    return json.loads(history_path.read_text())


def _check_learning(
    validation_loss: list[float], *, min_relative_improvement: float
) -> list[str]:
    reasons = []
    if len(validation_loss) < 2:  # noqa: PLR2004
        reasons.append("Not enough validation epochs recorded to assess learning.")
        return reasons

    first, last = validation_loss[0], validation_loss[-1]
    if first <= 0:
        reasons.append(
            f"Initial validation loss is non-positive ({first}); cannot assess "
            f"relative improvement."
        )
        return reasons

    improvement = (first - last) / first
    if improvement < min_relative_improvement:
        reasons.append(
            f"Validation loss only improved by {improvement:.1%} "
            f"(first={first:.4g}, last={last:.4g}); expected at least "
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
        output_dir: Directory to write the generated dataset, checkpoints, and report to.
        max_epochs: Optional override for ``train.trainer.max_epochs``.
        min_relative_improvement: Minimum fractional drop in validation loss (first
            epoch to last) required for the check to pass.

    Returns:
        A `SyntheticCheckResult` describing whether the check passed and why not.

    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Work on a detached copy so we never mutate the caller's config in place.
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    config["base_path"] = str(output_dir)
    if max_epochs is not None:
        config["train"]["trainer"]["max_epochs"] = max_epochs

    _generate_dataset(config, output_dir)

    service = ModelService.from_config(config)
    service.train()
    service.evaluate()

    history = _load_loss_history(output_dir / "report" / "loss_history.json")
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
