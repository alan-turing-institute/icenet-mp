from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import pytest
from omegaconf import DictConfig

DATA_DIR = Path(str(files("icenet_mp.config"))) / "data"
DATA_GROUP_CONFIGS = sorted(
    p.stem for p in DATA_DIR.glob("*.yaml") if not p.name.endswith(".local.yaml")
)


class TestDataConfigs:
    """Regression tests for icenet-mp's top-level data= config groups."""

    @pytest.mark.parametrize("config_name", DATA_GROUP_CONFIGS)
    def test_data_groups_compose(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        config = compose_config("sample", overrides=[f"data={config_name}"])

        assert config.data.datasets
        assert config.data.split.train
        assert config.data.split.test
        assert config.data.split.validate

    @pytest.mark.parametrize("config_name", ["full_north", "full_south"])
    def test_full_dataset_statistics_stop_at_training_boundary(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        """Full datasets must not use held-out years for normalisation stats."""
        config = compose_config("sample", overrides=[f"data={config_name}"])

        training_end = str(config.data.split.train[-1].end)
        held_out_starts = [
            str(period.start)
            for split_name in ("validate", "test", "predict")
            for period in config.data.split[split_name]
            if period.start is not None
        ]
        earliest_held_out = min(held_out_starts)

        assert training_end < earliest_held_out
        for dataset in config.data.datasets.values():
            assert dataset.statistics.end is not None
            statistics_end = str(dataset.statistics.end)
            assert statistics_end == training_end
            assert statistics_end < earliest_held_out
