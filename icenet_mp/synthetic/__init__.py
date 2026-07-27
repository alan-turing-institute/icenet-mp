from importlib import import_module

from .shapes import (
    GrowShrinkCircleConfig,
    MovingCircleConfig,
    generate_grow_shrink_frames,
    generate_moving_circle_frames,
)


def __getattr__(name: str) -> object:
    """Lazily import the pipeline API to avoid source-registration cycles."""
    if name in {
        "DYNAMICS_CHOICES",
        "DYNAMICS_GROW_SHRINK",
        "DYNAMICS_MOVING",
        "SyntheticCheckResult",
        "run_synthetic_pipeline_check",
    }:
        return getattr(import_module(".pipeline_check", __name__), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "DYNAMICS_CHOICES",
    "DYNAMICS_GROW_SHRINK",
    "DYNAMICS_MOVING",
    "GrowShrinkCircleConfig",
    "MovingCircleConfig",
    "SyntheticCheckResult",
    "generate_grow_shrink_frames",
    "generate_moving_circle_frames",
    "run_synthetic_pipeline_check",
]
