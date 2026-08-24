from importlib.resources import files

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

BASELINE_CONFIGS = [
    "00_persistence",
    "01_unet",
    "02_cnn_unet_cnn",
    "03_cnn_vit_cnn",
    "04_ddpm",
    "05_piecewise_unet_piecewise",
]


class TestBaselineConfigs:
    CONFIG_DIR = str(files("icenet_mp.config"))

    def setup_method(self) -> None:
        GlobalHydra.instance().clear()

    def teardown_method(self) -> None:
        GlobalHydra.instance().clear()

    @pytest.mark.parametrize("config_name", BASELINE_CONFIGS)
    def test_numbered_baseline_configs_compose(self, config_name: str) -> None:
        with initialize_config_dir(config_dir=self.CONFIG_DIR, version_base=None):
            config = compose(config_name=f"baseline/{config_name}")

        assert config.model.name
        assert config.model._target_.startswith("icenet_mp.models.")
        assert "data" in config
        assert "predict" in config
        assert "train" in config
        assert "evaluate" in config

    @pytest.mark.parametrize("config_name", BASELINE_CONFIGS)
    def test_numbered_baselines_accept_standard_data_override(
        self, config_name: str
    ) -> None:
        with initialize_config_dir(config_dir=self.CONFIG_DIR, version_base=None):
            config = compose(
                config_name=f"baseline/{config_name}",
                overrides=["data=sample_north"],
            )

        assert config.data.split.train
        assert config.data.split.test
        assert config.data.split.validate
