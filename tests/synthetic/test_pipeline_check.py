"""Tests for the pure pipeline-check helpers (no training involved)."""

import datetime

import pytest

from icenet_mp.synthetic.pipeline_check import (
    DYNAMICS_GROW_SHRINK,
    DYNAMICS_MOVING,
    _check_learning,
    _make_trajectories,
    _split_ranges,
    _validate_grid_size,
)
from icenet_mp.synthetic.shapes import GrowShrinkCircleConfig, MovingCircleConfig
from icenet_mp.synthetic.trajectories import TrajectorySpan


class TestValidateGridSize:
    """The grid size must satisfy UNetProcessor's divisible-by-16 constraint."""

    @pytest.mark.parametrize("grid_size", [32, 48, 144, 432])
    def test_accepts_valid(self, grid_size: int) -> None:
        """Multiples of 16 greater than 16 are accepted."""
        _validate_grid_size(grid_size)  # does not raise

    @pytest.mark.parametrize("grid_size", [0, 16, 17, 24, 40])
    def test_rejects_invalid(self, grid_size: int) -> None:
        """Values <= 16 or not divisible by 16 are rejected."""
        with pytest.raises(ValueError, match="divisible by 16"):
            _validate_grid_size(grid_size)


class TestSplitRanges:
    """Whole trajectories are assigned to train/validate/test."""

    def _spans(self, n: int) -> list[TrajectorySpan]:
        base = datetime.datetime(2020, 1, 1)
        return [
            TrajectorySpan(
                start_date=base + datetime.timedelta(days=10 * i),
                end_date=base + datetime.timedelta(days=10 * i + 9),
            )
            for i in range(n)
        ]

    def test_last_two_are_validate_and_test(self) -> None:
        """All but the final two spans train; the last two are validate then test."""
        ranges = _split_ranges(self._spans(5))
        assert len(ranges["train"]) == 3
        assert len(ranges["validate"]) == 1
        assert len(ranges["test"]) == 1

    def test_date_format(self) -> None:
        """Ranges are serialised as YYYY-MM-DD start/end pairs."""
        ranges = _split_ranges(self._spans(3))
        assert set(ranges["train"][0]) == {"start", "end"}
        assert ranges["train"][0]["start"] == "2020-01-01"

    def test_too_few_spans_raises(self) -> None:
        """Fewer than three trajectories cannot be split three ways."""
        with pytest.raises(ValueError, match="at least 3 trajectories"):
            _split_ranges(self._spans(2))


class TestCheckLearning:
    """The learning gate compares the best validation loss to the first epoch."""

    def test_sufficient_improvement_passes(self) -> None:
        """A large enough drop from the first epoch yields no failure reasons."""
        assert _check_learning([1.0, 0.5, 0.4], min_relative_improvement=0.3) == []

    def test_insufficient_improvement_fails(self) -> None:
        """Too small a drop is reported as a reason."""
        reasons = _check_learning([1.0, 0.95], min_relative_improvement=0.3)
        assert reasons
        assert "only improved" in reasons[0]

    def test_uses_best_not_last(self) -> None:
        """Overfitting after a good minimum still passes (best epoch is used)."""
        assert _check_learning([1.0, 0.2, 0.9], min_relative_improvement=0.3) == []

    def test_too_few_epochs_fails(self) -> None:
        """A single validation epoch cannot demonstrate learning."""
        reasons = _check_learning([1.0], min_relative_improvement=0.3)
        assert reasons
        assert "Not enough" in reasons[0]

    def test_non_positive_first_loss_fails(self) -> None:
        """A non-positive initial loss makes relative improvement undefined."""
        reasons = _check_learning([0.0, -1.0], min_relative_improvement=0.3)
        assert reasons
        assert "non-positive" in reasons[0]


class TestMakeTrajectories:
    """`_make_trajectories` selects the generator for the requested dynamics."""

    def test_moving(self) -> None:
        """'moving' yields moving-circle configs."""
        configs = _make_trajectories(
            dynamics=DYNAMICS_MOVING, grid_size=32, n_trajectories=4
        )
        assert all(isinstance(c, MovingCircleConfig) for c in configs)

    def test_grow_shrink(self) -> None:
        """'grow-shrink' yields grow-shrink configs."""
        configs = _make_trajectories(
            dynamics=DYNAMICS_GROW_SHRINK, grid_size=48, n_trajectories=4
        )
        assert all(isinstance(c, GrowShrinkCircleConfig) for c in configs)

    def test_unknown_dynamics_raises(self) -> None:
        """An unrecognised dynamics name is rejected."""
        with pytest.raises(ValueError, match="Unknown dynamics"):
            _make_trajectories(dynamics="spinning", grid_size=32, n_trajectories=4)
