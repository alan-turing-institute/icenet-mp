import inspect
from importlib.resources import files

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from icenet_mp.cli.hydra import hydra_adaptor


class TestHydraConfigLoading:
    """Regression tests for icenet-mp config composition via hydra."""

    CONFIG_DIR = str(files("icenet_mp.config"))

    def setup_method(self) -> None:
        GlobalHydra.instance().clear()

    def teardown_method(self) -> None:
        GlobalHydra.instance().clear()

    def load_config(
        self, config_name: str = "sample", overrides: list[str] | None = None
    ) -> DictConfig:
        """Compose a config from the icenet-mp config directory with any overrides."""
        with initialize_config_dir(config_dir=self.CONFIG_DIR, version_base=None):
            return compose(config_name=config_name, overrides=overrides or [])

    def test_sample_config_has_expected_top_level_keys(self) -> None:
        cfg = self.load_config()
        for key in ("data", "model", "train", "loss", "predict", "evaluate"):
            assert key in cfg, f"Key '{key}' missing from composed config"

    def test_model_group_overridden_by_sample(self) -> None:
        # sample.yaml uses `override /model: quick_test`, replacing the base default
        cfg = self.load_config()
        assert cfg.model.name == "quick-test"
        assert cfg.model._target_ == "icenet_mp.models.EncodeProcessDecode"

    def test_loss_defaults_resolved_from_base(self) -> None:
        cfg = self.load_config()
        assert cfg.loss._target_ == "icenet_mp.losses.amse_loss.AMSELoss"
        assert cfg.loss.delta == pytest.approx(0.5)

    def test_scalar_override_applied(self) -> None:
        cfg = self.load_config(overrides=["loss.delta=1.0"])
        assert cfg.loss.delta == pytest.approx(1.0)

    def test_wandb_offline_override(self) -> None:
        cfg = self.load_config()
        assert cfg.loggers.wandb.offline is False

        cfg = self.load_config(overrides=["loggers.wandb.offline=true"])
        assert cfg.loggers.wandb.offline is True

    def test_config_group_override_swaps_loss(self) -> None:
        cfg = self.load_config(overrides=["loss=mse"])
        assert cfg.loss._target_ == "torch.nn.MSELoss"

    def test_synthetic_config_uses_local_logger(self) -> None:
        cfg = self.load_config(config_name="synthetic")
        assert "local_files" in cfg.loggers
        assert "wandb" not in cfg.loggers
        assert cfg.loggers.local_files._target_ == "icenet_mp.loggers.LocalFileLogger"
        assert "metric_summary" not in cfg.train.callbacks
        assert "metric_summary" not in cfg.evaluate.callbacks

    def test_piecewise_baselines_are_matched_except_for_model_variant(self) -> None:
        baseline = self.load_config(config_name="baseline/05_piecewise_unet_piecewise")
        naive = self.load_config(
            config_name="baseline/06_piecewise_unet_piecewise_naive"
        )

        assert baseline.model.name == "piecewise-unet-piecewise"
        assert naive.model.name == "piecewise-unet-piecewise-naive"
        for key in ("data", "loss", "predict", "train", "evaluate", "random"):
            assert baseline[key] == naive[key]


class TestHydraAdaptor:
    """Regression tests for icenet-mp's hydra_adaptor signature rewriter."""

    def test_signature_rewriting(self) -> None:
        def fn(config: DictConfig) -> None:
            pass

        params = inspect.signature(hydra_adaptor(fn)).parameters
        assert "config" not in params
        assert "config_name" in params
        assert "overrides" in params

    def test_preserves_positional_params(self) -> None:
        def fn(x: int, config: DictConfig) -> None:
            del x, config

        params = list(inspect.signature(hydra_adaptor(fn)).parameters)
        assert "x" in params
        assert "config" not in params

    def test_preserves_keyword_only_params(self) -> None:
        def fn(config: DictConfig, *, flag: bool = False) -> None:
            del config, flag

        params = inspect.signature(hydra_adaptor(fn)).parameters
        assert "flag" in params
        assert params["flag"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_preserves_name_and_doc(self) -> None:
        def fn(config: DictConfig) -> None:
            """My docstring."""

        assert hydra_adaptor(fn).__name__ == "fn"
        assert hydra_adaptor(fn).__doc__ == "My docstring."

    def test_wrapped_function_receives_dictconfig(self) -> None:
        received: list[DictConfig] = []

        def fn(config: DictConfig) -> None:
            received.append(config)

        hydra_adaptor(fn)(config_name="sample")  # type: ignore[arg-type]
        assert len(received) == 1
        assert isinstance(received[0], DictConfig)

    def test_wrapped_function_forwards_kwargs(self) -> None:
        received: list[tuple] = []

        def fn(x: int, config: DictConfig) -> None:
            received.append((x, config))

        hydra_adaptor(fn)(x=42, config_name="sample")  # type: ignore[arg-type]
        assert received[0][0] == 42
        assert isinstance(received[0][1], DictConfig)
