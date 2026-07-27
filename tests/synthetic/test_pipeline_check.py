"""Tests for the pure pipeline-check helpers (no training involved)."""

import json
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from icenet_mp.synthetic.pipeline_check import (
    _check_learning,
    _clear_loss_history,
    _load_loss_history,
    run_synthetic_pipeline_check,
)


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


class TestLoadLossHistory:
    """Loss curves reuse the local metrics logger output."""

    def test_reads_train_and_validation_losses(self, tmp_path: Path) -> None:
        """Extract epoch losses while ignoring unrelated local metrics."""
        metrics_path = tmp_path / "metrics.jsonl"
        metrics_path.write_text(
            "\n".join(
                [
                    json.dumps({"train_loss": 0.8, "step": 1}),
                    json.dumps({"validation_loss": 0.6, "step": 1}),
                    json.dumps({"test_loss": 0.5, "step": 2}),
                ]
            )
            + "\n"
        )

        assert _load_loss_history(metrics_path) == {
            "train_loss": [0.8],
            "validation_loss": [0.6],
        }

    def test_clears_existing_history(self, tmp_path: Path) -> None:
        """A reused output directory must not retain metrics from an earlier run."""
        metrics_path = tmp_path / "metrics.jsonl"
        metrics_path.write_text('{"validation_loss": 0.9}\n')

        _clear_loss_history(metrics_path)

        assert not metrics_path.exists()
        _clear_loss_history(metrics_path)  # Missing history is also safe.


def test_check_uses_configured_dataset_base_path(tmp_path: Path) -> None:
    """The check must consume the dataset created by the separate CLI command."""
    (tmp_path / "report").mkdir()
    config = OmegaConf.create(
        {
            "base_path": "/created-dataset",
            "train": {"trainer": {}},
            "model": {},
        }
    )
    with (
        patch("icenet_mp.synthetic.pipeline_check.ModelService.from_config") as from_config,
        patch(
            "icenet_mp.synthetic.pipeline_check._load_loss_history",
            return_value={"train_loss": [1.0], "validation_loss": [1.0, 0.5]},
        ),
        patch("icenet_mp.synthetic.pipeline_check.plot_loss_curve"),
    ):
        from_config.return_value.train.return_value = None
        from_config.return_value.evaluate.return_value = None

        run_synthetic_pipeline_check(config, output_dir=tmp_path)

    assert from_config.call_args.args[0]["base_path"] == "/created-dataset"
