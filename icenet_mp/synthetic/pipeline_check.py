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
from icenet_mp.types import ArrayTHW

from .debug_video import write_full_dataset_video, write_full_rollout_video
from .report import plot_loss_curve
from .trajectories import (
    TrajectorySpan,
    default_grow_shrink_configs,
    default_trajectory_configs,
    generate_multi_trajectory_dataset,
)
from .zarr_writer import write_synthetic_zarr

logger = logging.getLogger(__name__)

SYNTHETIC_DATASET_NAME = "synthetic-sic"
SYNTHETIC_VARIABLE_NAME = "ice_conc"

# Selectable synthetic dynamics: a translating circle (advection) or a stationary blob
# that grows and shrinks in place (morphological open/close, mimicking seasonal ice
# advance/retreat).
DYNAMICS_MOVING = "moving"
DYNAMICS_GROW_SHRINK = "grow-shrink"
DYNAMICS_CHOICES = (DYNAMICS_MOVING, DYNAMICS_GROW_SHRINK)


@dataclass
class SyntheticCheckResult:
    passed: bool
    reasons: list[str]
    train_loss: list[float]
    validation_loss: list[float]
    report_path: Path


def _date_range(span: TrajectorySpan) -> dict[str, str]:
    return {
        "start": span.start_date.strftime("%Y-%m-%d"),
        "end": span.end_date.strftime("%Y-%m-%d"),
    }


def _split_ranges(spans: list[TrajectorySpan]) -> dict[str, list[dict[str, str]]]:
    """Assign whole trajectories to train/validate/test.

    Splitting by trajectory (not by day range within one trajectory) means validation
    and test see (start position, velocity) combinations the model never trained on,
    so passing the check requires learning the general update rule, not memorising
    specific days.
    """
    if len(spans) < 3:  # noqa: PLR2004
        msg = f"Need at least 3 trajectories to split train/validate/test, got {len(spans)}."
        raise ValueError(msg)
    *train_spans, validate_span, test_span = spans
    return {
        "train": [_date_range(span) for span in train_spans],
        "validate": [_date_range(validate_span)],
        "test": [_date_range(test_span)],
    }


def _validate_grid_size(grid_size: int) -> None:
    """UNetProcessor requires latent height/width each divisible by 16 and > 16."""
    if grid_size <= 16 or grid_size % 16:  # noqa: PLR2004
        msg = (
            f"grid_size must be > 16 and divisible by 16 (UNetProcessor's own "
            f"constraint), got {grid_size}."
        )
        raise ValueError(msg)


def _make_trajectories(*, dynamics: str, grid_size: int, n_trajectories: int) -> tuple:
    if dynamics == DYNAMICS_MOVING:
        return default_trajectory_configs(
            height=grid_size, width=grid_size, n_trajectories=n_trajectories
        )
    if dynamics == DYNAMICS_GROW_SHRINK:
        return default_grow_shrink_configs(
            height=grid_size, width=grid_size, n_trajectories=n_trajectories
        )
    msg = f"Unknown dynamics {dynamics!r}; expected one of {DYNAMICS_CHOICES}."
    raise ValueError(msg)


def _generate_dataset(
    config: DictConfig,
    output_dir: Path,
    *,
    grid_size: int,
    n_trajectories: int,
    dynamics: str,
) -> tuple[Path, ArrayTHW, list, dict[str, list[dict[str, str]]]]:
    dataset_entries = list(config["data"]["datasets"].values())
    unknown = [
        entry["name"]
        for entry in dataset_entries
        if entry["name"] != SYNTHETIC_DATASET_NAME
    ]
    if unknown:
        msg = (
            f"Don't know how to generate synthetic data for dataset(s) {unknown}. "
            f"The synthetic pipeline check currently only supports a single dataset "
            f"named '{SYNTHETIC_DATASET_NAME}' -- use the 'data=synthetic' config group."
        )
        raise ValueError(msg)

    _validate_grid_size(grid_size)
    trajectories = _make_trajectories(
        dynamics=dynamics, grid_size=grid_size, n_trajectories=n_trajectories
    )
    dataset = generate_multi_trajectory_dataset(trajectories)
    zarr_path = output_dir / "data" / "anemoi" / f"{SYNTHETIC_DATASET_NAME}.zarr"
    write_synthetic_zarr(
        zarr_path,
        frames=dataset.frames,
        variable_name=SYNTHETIC_VARIABLE_NAME,
        missing_dates=dataset.missing_dates,
    )
    logger.info(
        "Wrote %d independent synthetic trajectories (%d days total) to %s.",
        len(dataset.spans),
        len(dataset.dates),
        zarr_path,
    )
    return zarr_path, dataset.frames, dataset.dates, _split_ranges(dataset.spans)


def _load_loss_history(history_path: Path) -> dict[str, list[float]]:
    if not history_path.exists():
        msg = (
            f"Expected loss history at {history_path}, but it does not exist. Does the "
            f"config include the 'loss_history' logger?"
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


def run_synthetic_pipeline_check(  # noqa: PLR0913
    config: DictConfig,
    *,
    output_dir: Path,
    max_epochs: int | None = None,
    min_relative_improvement: float = 0.3,
    dump_debug_video: bool = False,
    grid_size: int = 32,
    n_trajectories: int = 8,
    dynamics: str = DYNAMICS_MOVING,
) -> SyntheticCheckResult:
    """Run a model+data config against synthetic data and check that it learns.

    Args:
        config: A composed Hydra config, e.g. from the ``synthetic_unet`` baseline.
        output_dir: Directory to write the generated dataset, checkpoints, and report to.
        max_epochs: Optional override for ``train.trainer.max_epochs``.
        min_relative_improvement: Minimum fractional drop from the first epoch's
            validation loss to the best epoch's, required for the check to pass.
        dump_debug_video: If True, additionally render the entire generated dataset and
            a full-dataset ground-truth-vs-prediction rollout as videos under
            ``output_dir/report/debug``. Off by default: it re-runs inference across
            every window in the dataset and adds real wall-clock time.
        grid_size: Height/width of the synthetic grid. The model's `encoders.
            latent_space` is set to match automatically, so this does not need a
            separate model override. Must be > 16 and divisible by 16 (UNetProcessor's
            own constraint). Defaults to a small, fast 32; pass e.g. 432 to match real
            data's native resolution, at the cost of much longer training time.
        n_trajectories: Number of independent trajectories to generate (split: all but
            the last two for training, one for validation, one for test). More
            trajectories means more/more-diverse training data, at the cost of longer
            training time.
        dynamics: Which synthetic dynamics to generate. ``"moving"`` (default) is a
            circle translating and bouncing off the grid edges (advection);
            ``"grow-shrink"`` is a stationary blob that grows and shrinks in place via
            morphological open/close, mimicking seasonal ice advance/retreat.

    Returns:
        A `SyntheticCheckResult` describing whether the check passed and why not.

    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Work on a detached copy so we never mutate the caller's config in place.
    config = DictConfig(OmegaConf.to_container(config, resolve=False))
    config["base_path"] = str(output_dir)
    if max_epochs is not None:
        config["train"]["trainer"]["max_epochs"] = max_epochs
    config["model"]["encoders"]["latent_space"] = [grid_size, grid_size]

    zarr_path, frames, dates, split_ranges = _generate_dataset(
        config,
        output_dir,
        grid_size=grid_size,
        n_trajectories=n_trajectories,
        dynamics=dynamics,
    )
    config["data"]["split"]["train"] = split_ranges["train"]
    config["data"]["split"]["validate"] = split_ranges["validate"]
    config["data"]["split"]["test"] = split_ranges["test"]

    if dump_debug_video:
        write_full_dataset_video(
            frames=frames,
            dates=dates,
            variable_name=SYNTHETIC_VARIABLE_NAME,
            output_path=output_dir / "report" / "debug" / "full_dataset.mp4",
        )

    service = ModelService.from_config(config)
    service.train()
    service.evaluate()

    if dump_debug_video:
        write_full_rollout_video(
            model=service.model,
            zarr_path=zarr_path,
            target_group_name=config["predict"]["target"]["group_name"],
            target_variables=list(config["predict"]["target"].get("variables", [])),
            n_history_steps=int(config["predict"]["n_history_steps"]),
            n_forecast_steps=int(config["predict"]["n_forecast_steps"]),
            variable_name=SYNTHETIC_VARIABLE_NAME,
            output_path=output_dir / "report" / "debug" / "full_rollout.mp4",
        )

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
