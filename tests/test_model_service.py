from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from icenet_mp.model_service import ModelService
from icenet_mp.types import DataSpace


class MockCommonDataModule:
    def __init__(self, config: DictConfig) -> None:
        """Mock CommonDataModule."""
        self.config = config
        self.input_spaces = [DataSpace(5, "input", (20, 20))]
        self.n_forecast_steps = 2
        self.n_history_steps = 3
        self.output_space = DataSpace(1, "output", (10, 10))


class MockModel:
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str | Path) -> "MockModel":
        del checkpoint_path
        return cls()


class TestModelService:
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
        model_config = args[0]
        assert model_config["input_spaces"] == [
            DataSpace(5, "input", (20, 20)).to_dict()
        ]
        assert (
            model_config["output_space"] == DataSpace(1, "output", (10, 10)).to_dict()
        )
        assert model_config["n_forecast_steps"] == 2
        assert model_config["n_history_steps"] == 3
        assert model_config["optimizer"] is cfg_model_service["train"]["optimizer"]
        assert model_config["scheduler"] is cfg_model_service["train"]["scheduler"]
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
