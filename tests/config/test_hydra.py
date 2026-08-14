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
        assert cfg.loss._target_ == "torch.nn.HuberLoss"
        assert cfg.loss.delta == pytest.approx(0.5)

    def test_scalar_override_applied(self) -> None:
        cfg = self.load_config(overrides=["loss.delta=1.0"])
        assert cfg.loss.delta == pytest.approx(1.0)

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

    def test_feature_screening_policy_is_opt_in(self) -> None:
        """The screening baseline retains importance without changing older baselines."""
        existing = self.load_config(
            config_name="rf_screening/02_feature_evidence_registry"
        )
        screening = self.load_config(config_name="rf_screening/03_feature_screening")

        assert existing.rf.get("importance_policy", "qualified") == "qualified"
        assert screening.rf.importance_policy == "always"
        assert screening.rf.target_mode == "sic_change"

    def test_feature_screening_backend_selection_is_opt_in(self) -> None:
        """Default backend is RandomForest; HistGradientBoosting is opt-in via a new baseline."""
        existing = self.load_config(config_name="rf_screening/03_feature_screening")
        hgb = self.load_config(
            config_name="rf_screening/04_feature_screening_hist_gradient_boosting"
        )

        assert existing.rf.backend == "random_forest"
        assert hgb.rf.backend == "hist_gradient_boosting"
        # The opt-in baseline inherits every other screening setting verbatim.
        assert hgb.rf.importance_policy == existing.rf.importance_policy
        assert hgb.rf.target_mode == existing.rf.target_mode
        assert (
            hgb.rf.spatial.locations_per_stratum
            == existing.rf.spatial.locations_per_stratum
        )


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

    def test_config_name_not_declared_is_not_forwarded(self) -> None:
        """Functions that don't ask for config_name see no behaviour change."""

        def fn(config: DictConfig) -> None:
            del config

        params = inspect.signature(hydra_adaptor(fn)).parameters
        assert list(params) == ["overrides", "config_name"]

    def test_config_name_declared_is_forwarded_without_duplicate_cli_option(
        self,
    ) -> None:
        received: dict[str, object] = {}

        def fn(config: DictConfig, config_name: str = "sample") -> None:
            received["config"] = config
            received["config_name"] = config_name

        wrapped = hydra_adaptor(fn)
        # Only one config_name parameter should reach the CLI surface.
        assert list(inspect.signature(wrapped).parameters).count("config_name") == 1

        wrapped(config_name="synthetic")  # type: ignore[arg-type]
        assert received["config_name"] == "synthetic"
        assert isinstance(received["config"], DictConfig)
