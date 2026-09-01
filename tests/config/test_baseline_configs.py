from collections.abc import Callable

import pytest
from omegaconf import DictConfig

BASELINE_CONFIGS = [
    "00_persistence",
    "01_unet",
    "02_cnn_unet_cnn",
    "03_cnn_vit_cnn",
    "04_ddpm",
    "05_piecewise_unet_piecewise",
]


class TestBaselineConfigs:
    """Regression tests for icenet-mp baseline configs."""

    @pytest.mark.parametrize("config_name", BASELINE_CONFIGS)
    def test_numbered_baseline_configs_compose(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        config = compose_config(f"baseline/{config_name}")

        assert config.model.name
        assert config.model._target_.startswith("icenet_mp.models.")
        assert "data" in config
        assert "predict" in config
        assert "train" in config
        assert "evaluate" in config

    @pytest.mark.parametrize("config_name", BASELINE_CONFIGS)
    def test_numbered_baselines_accept_standard_data_override(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        config = compose_config(
            f"baseline/{config_name}", overrides=["data=sample_north"]
        )

        assert config.data.split.train
        assert config.data.split.test
        assert config.data.split.validate
