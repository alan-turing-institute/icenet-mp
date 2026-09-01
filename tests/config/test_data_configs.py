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
