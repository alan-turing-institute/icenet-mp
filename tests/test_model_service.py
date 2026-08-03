from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from icenet_mp import model_service
from icenet_mp.model_service import ModelService
from icenet_mp.types import DataSpace


class MockCommonDataModule:
    def __init__(self, config: DictConfig) -> None:
        """Mock CommonDataModule."""
        self.config = config
        self.hemisphere = "north"
        self.input_spaces = [DataSpace(5, "input", (20, 20))]
        self.latitudes = {"input": [0.0] * 400}
        self.longitudes = {"input": [0.0] * 400}
        self.mask_directory = Path("nonexistent")
        self.n_forecast_steps = 2
        self.n_history_steps = 3
        self.output_space = DataSpace(1, "output", (10, 10))
        self.target_variable_indices = [0]


class MockModel:
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        mask_dir: str | None = None,
        latitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        longitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        map_location: str | None = None,
        weights_only: bool = False,
    ) -> "MockModel":
        del (
            mask_dir,
            checkpoint_path,
            latitudes_fn,
            longitudes_fn,
            map_location,
            weights_only,
        )
        return cls()


class TestModelService:
    def test_recovers_legacy_vit_decode_head(self, tmp_path: Path) -> None:
        """Legacy enhanced ViT checkpoints retain their omitted constructor defaults."""
        checkpoint_path = tmp_path / "legacy.ckpt"
        torch.save(
            {
                "hyper_parameters": {
                    "processor": {
                        "_target_": "icenet_mp.models.processors.VitProcessor",
                        "patch_size": 4,
                    }
                },
                "state_dict": {
                    "processor.patch_to_pixels.weight": torch.empty(128, 8),
                    "processor.refine.0.block.0.block.0.weight": torch.empty(
                        8, 8, 5, 5
                    ),
                },
            },
            checkpoint_path,
        )

        overrides = model_service._checkpoint_constructor_overrides(checkpoint_path)

        processor = overrides["processor"]
        assert isinstance(processor, DictConfig)
        assert processor.decode_head == "conv_refine"
        assert processor.refine_channels == 8
        assert processor.refine_kernel_size == 5

    def test_from_config_loads_model(self, cfg_model_service: DictConfig) -> None:
        mock_instantiate = MagicMock()
        mock_instantiate.return_value = MockModel()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("icenet_mp.model_service.CommonDataModule", MockCommonDataModule)
            mp.setattr(
                "icenet_mp.model_service.hydra.utils.instantiate", mock_instantiate
            )
            service = ModelService.from_config(cfg_model_service)
            assert isinstance(service.model, MockModel)

        args, kwargs = mock_instantiate.call_args
        assert args[0] is cfg_model_service["model"]
        assert kwargs["input_spaces"] == [DataSpace(5, "input", (20, 20)).to_dict()]
        assert kwargs["output_space"] == DataSpace(1, "output", (10, 10)).to_dict()
        assert kwargs["n_forecast_steps"] == 2
        assert kwargs["n_history_steps"] == 3
        assert kwargs["optimizer"] is cfg_model_service["train"]["optimizer"]
        assert kwargs["scheduler"] is cfg_model_service["train"]["scheduler"]
        assert kwargs["_recursive_"] is False
        assert kwargs["_convert_"] == "object"

    def test_from_checkpoint_loads_model(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        # Generate a checkpoint file and corresponding model_config.yaml
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "model.ckpt"
        checkpoint_path.write_text("checkpoint")

        files_dir = tmp_path / "files"
        files_dir.mkdir(parents=True)
        OmegaConf.save(cfg_model_service, files_dir / "model_config.yaml")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("icenet_mp.model_service.CommonDataModule", MockCommonDataModule)
            mp.setattr(
                "icenet_mp.model_service.hydra.utils.get_class",
                lambda _target: MockModel,
            )
            service = ModelService.from_checkpoint(DictConfig({}), checkpoint_path)
            assert isinstance(service.model, MockModel)
            assert service.config == cfg_model_service

    def test_from_checkpoint_config_overloads(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        # Generate a checkpoint file and corresponding model_config.yaml
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "model.ckpt"
        checkpoint_path.write_text("checkpoint")

        files_dir = tmp_path / "files"
        files_dir.mkdir(parents=True)
        OmegaConf.save(cfg_model_service, files_dir / "model_config.yaml")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("icenet_mp.model_service.CommonDataModule", MockCommonDataModule)
            mp.setattr(
                "icenet_mp.model_service.hydra.utils.get_class",
                lambda _target: MockModel,
            )
            service = ModelService.from_checkpoint(
                DictConfig(
                    {
                        "loggers": "will_overwrite",
                        "model": {"name": "will_not_overwrite"},
                    }
                ),
                checkpoint_path,
            )
            assert isinstance(service.model, MockModel)

            expected_config = cfg_model_service.copy()
            expected_config["loggers"] = "will_overwrite"
            assert service.config == expected_config
            assert service.config["model"]["name"] != "will_not_overwrite"

    def test_isolated_evaluation_replaces_saved_callbacks_and_loggers(
        self, cfg_model_service: DictConfig, tmp_path: Path
    ) -> None:
        """Do not resurrect stale checkpoint integrations for isolated evaluation."""
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "model.ckpt"
        checkpoint_path.write_text("checkpoint")
        saved_config = cfg_model_service.copy()
        saved_config.evaluate.callbacks = {
            "activation_saver": {"_target_": "saved.ActivationSaver"},
            "metric_summary": {"_target_": "saved.MetricSummaryCallback"},
            "plotting": {"_target_": "saved.PlottingCallback"},
        }
        saved_config.loggers = {
            "wandb": {"_target_": "lightning.pytorch.loggers.WandbLogger"}
        }
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        OmegaConf.save(saved_config, files_dir / "model_config.yaml")
        current_config = DictConfig(
            {
                "evaluate": {
                    "callbacks": {
                        "isolated_evaluation": {
                            "_target_": "icenet_mp.callbacks.IsolatedEvaluationCallback"
                        }
                    }
                },
                "loggers": {
                    "isolated_evaluation": {
                        "_target_": "icenet_mp.loggers.LocalFileLogger"
                    }
                },
            }
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("icenet_mp.model_service.CommonDataModule", MockCommonDataModule)
            mp.setattr(
                "icenet_mp.model_service.hydra.utils.get_class",
                lambda _target: MockModel,
            )
            service = ModelService.from_checkpoint(current_config, checkpoint_path)

        assert list(service.config.evaluate.callbacks) == ["isolated_evaluation"]
        assert list(service.config.loggers) == ["isolated_evaluation"]
