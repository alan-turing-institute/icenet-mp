from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

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
        assert kwargs["lr_scheduler"] is cfg_model_service["train"]["lr_scheduler"]
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

    def test_train_standard_mode_rejects_model_requiring_multistage(self) -> None:
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock()
        service.model_.multistage_only = True

        with pytest.raises(ValueError, match="multistage"):
            service.train()

    def test_train_standard_mode_allows_model_not_requiring_multistage(
        self,
    ) -> None:
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock()
        service.model_.multistage_only = False
        service.config_ = DictConfig({"train": "train_config"})

        with pytest.MonkeyPatch.context() as mp:
            mock_fit = MagicMock()
            mp.setattr(service, "_fit", mock_fit)
            service.train()

        mock_fit.assert_called_once_with(config="train_config", ckpt_path=None)

    def test_merged_config_applies_stage_overrides(self) -> None:
        """Merge stage-specific values over the common training configuration."""
        service = ModelService.__new__(ModelService)
        service.config_ = OmegaConf.create(
            {
                "train": {
                    "optimizer": {"lr": 0.001, "weight_decay": 0.01},
                    "trainer": {"max_epochs": 20, "accelerator": "auto"},
                    "multistage": {
                        "processor": {
                            "optimizer": {"lr": 0.01},
                            "trainer": {"max_epochs": 3},
                        }
                    },
                }
            }
        )

        merged = service._merged_config("processor")

        assert merged["optimizer"]["lr"] == 0.01
        assert merged["optimizer"]["weight_decay"] == 0.01
        assert merged["trainer"]["max_epochs"] == 3
        assert merged["trainer"]["accelerator"] == "auto"

    def test_save_stage_checkpoint_saves_when_no_best_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """Save a deterministic stage checkpoint when callbacks have no best path."""
        service = ModelService.__new__(ModelService)
        trainer = MagicMock()
        trainer.current_epoch = 4
        trainer.global_step = 17
        trainer.checkpoint_callbacks = []
        trainer.is_global_zero = True
        run_dir = tmp_path / "run"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service, "build_run_directory", lambda _trainer: run_dir)
            result = service._save_stage_checkpoint(trainer, "processor")

        expected = run_dir / "checkpoints" / "processor.epoch=4-step=17.ckpt"
        trainer.save_checkpoint.assert_called_once_with(expected, weights_only=False)
        assert result == expected

    def test_save_stage_checkpoint_moves_single_best_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """Move one callback-selected best checkpoint to the stage checkpoint path."""
        service = ModelService.__new__(ModelService)
        run_dir = tmp_path / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        best_path = tmp_path / "best.ckpt"
        best_path.write_text("checkpoint")

        trainer = MagicMock()
        trainer.current_epoch = 2
        trainer.global_step = 8
        trainer.is_global_zero = True
        trainer.checkpoint_callbacks = [SimpleNamespace(best_model_path=str(best_path))]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service, "build_run_directory", lambda _trainer: run_dir)
            result = service._save_stage_checkpoint(trainer, "decoder")

        expected = run_dir / "checkpoints" / "decoder.best.ckpt"
        assert result == expected
        assert expected.read_text() == "checkpoint"
        trainer.save_checkpoint.assert_not_called()
        trainer.strategy.barrier.assert_called_once_with()

    def test_save_stage_checkpoint_rejects_multiple_best_paths(
        self, tmp_path: Path
    ) -> None:
        """Reject ambiguous checkpoint selection from multiple callbacks."""
        service = ModelService.__new__(ModelService)
        trainer = MagicMock()
        trainer.current_epoch = 1
        trainer.global_step = 2
        trainer.checkpoint_callbacks = [
            SimpleNamespace(best_model_path=str(tmp_path / "a.ckpt")),
            SimpleNamespace(best_model_path=str(tmp_path / "b.ckpt")),
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service, "build_run_directory", lambda _trainer: tmp_path)
            with pytest.raises(ValueError, match="2 checkpoints"):
                service._save_stage_checkpoint(trainer, "encoder")

    def test_train_standard_mode_rejects_checkpoint_dir_without_last_ckpt(
        self, tmp_path: Path
    ) -> None:
        """Reject a checkpoint directory with no resumable ``last*.ckpt`` file."""
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock()
        service.model_.multistage_only = False
        service.config_ = DictConfig({"train": "train_config"})

        with pytest.raises(FileNotFoundError, match=r"last\*.ckpt"):
            service.train(checkpoint_dir=tmp_path)

    def test_train_standard_mode_resumes_from_last_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """Resume single-stage training from the ``last*.ckpt`` file if present."""
        service = ModelService.__new__(ModelService)
        service.model_ = MagicMock()
        service.model_.multistage_only = False
        service.config_ = DictConfig({"train": "train_config"})
        ckpt_path = tmp_path / "last.ckpt"
        ckpt_path.write_text("checkpoint")

        with pytest.MonkeyPatch.context() as mp:
            mock_fit = MagicMock()
            mp.setattr(service, "_fit", mock_fit)
            service.train(checkpoint_dir=tmp_path)

        mock_fit.assert_called_once_with(config="train_config", ckpt_path=ckpt_path)
