from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from icenet_mp.losses.rmse_loss import RMSELoss
from icenet_mp.models import BaseModel
from icenet_mp.types import ModelTestOutput, TensorNTCHW


class FakeDataModel(BaseModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise a fake data model for testing purposes."""
        super().__init__(*args, hemisphere="north", **kwargs)
        self.t = kwargs["n_forecast_steps"]
        self.c = kwargs["output_space"]["channels"]
        self.h = kwargs["output_space"]["shape"][0]
        self.w = kwargs["output_space"]["shape"][1]
        self.model = torch.nn.Linear(1, 1)

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """FakeData forward method."""
        b = next(iter(inputs.values())).shape[0]
        return torch.randn(b, self.t, self.c, self.h, self.w)

_DEFAULT_LOSS = OmegaConf.create({"_target_": "torch.nn.HuberLoss", "delta": 0.5})

class TestBaseModel:
    @pytest.mark.parametrize("test_input_chw", [(4, 512, 512), (1, 10, 20)])
    @pytest.mark.parametrize("test_output_chw", [(1, 432, 432), (19, 10, 20)])
    @pytest.mark.parametrize("test_n_forecast_steps", [0, 1, 2, 5])
    @pytest.mark.parametrize("test_n_history_steps", [0, 1, 2, 5])
    def test_init(
        self,
        test_input_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_output_chw: tuple[int, int, int],
    ) -> None:
        input_space = DictConfig(
            {
                "channels": test_input_chw[0],
                "name": "input",
                "shape": test_input_chw[1:],
            }
        )
        output_space = DictConfig(
            {
                "channels": test_output_chw[0],
                "name": "target",
                "shape": test_output_chw[1:],
            }
        )

        # Catch invalid n_forecast_steps
        if test_n_forecast_steps <= 0:
            with pytest.raises(
                ValueError, match="Number of forecast steps must be greater than 0."
            ):
                FakeDataModel(
                    name="fake data",
                    input_spaces=[input_space],
                    n_forecast_steps=test_n_forecast_steps,
                    n_history_steps=test_n_history_steps,
                    output_space=output_space,
                    optimizer=DictConfig({}),
                    scheduler=DictConfig({}),
                    loss=_DEFAULT_LOSS,
                )
            return

        # Catch invalid n_history_steps
        if test_n_history_steps <= 0:
            with pytest.raises(
                ValueError, match="Number of history steps must be greater than 0."
            ):
                FakeDataModel(
                    name="fake data",
                    input_spaces=[input_space],
                    n_forecast_steps=test_n_forecast_steps,
                    n_history_steps=test_n_history_steps,
                    output_space=output_space,
                    optimizer=DictConfig({}),
                    scheduler=DictConfig({}),
                )
            return

        model = FakeDataModel(
            name="fake data",
            input_spaces=[input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            loss=_DEFAULT_LOSS,
        )

        assert model.name == "fake data"
        assert model.input_spaces[0].channels == test_input_chw[0]
        assert model.input_spaces[0].name == "input"
        assert model.input_spaces[0].shape == test_input_chw[1:]
        assert model.n_forecast_steps == test_n_forecast_steps
        assert model.n_history_steps == test_n_history_steps
        assert model.output_space.channels == test_output_chw[0]
        assert model.output_space.name == "target"
        assert model.output_space.shape == test_output_chw[1:]

    def test_loss(
        self, cfg_input_space: DictConfig, cfg_output_space: DictConfig
    ) -> None:
        model = FakeDataModel(
            name="fake data",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            loss=_DEFAULT_LOSS,
        )
        # Test loss
        prediction = torch.zeros(1, 1, 1, 1)
        target = torch.ones(1, 1, 1, 1)
        # assert model.loss(prediction, target) == torch.tensor(1.0)
        result = model.loss(prediction, target)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result.item() > 0

    def test_optimizer(
        self,
        cfg_input_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        model = FakeDataModel(
            name="fake data",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=cfg_optimizer,
            scheduler=DictConfig({}),
            loss=_DEFAULT_LOSS,
        )
        opt_sched_cfg = model.configure_optimizers()
        assert isinstance(opt_sched_cfg, dict)
        optimizer = opt_sched_cfg.get("optimizer", None)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 5e-4

    def test_scheduler(
        self,
        cfg_input_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_output_space: DictConfig,
        cfg_scheduler: DictConfig,
    ) -> None:
        model = FakeDataModel(
            name="dummy",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=cfg_optimizer,
            scheduler=cfg_scheduler,
            loss=_DEFAULT_LOSS,
        )
        opt_sched_cfg = model.configure_optimizers()
        assert isinstance(opt_sched_cfg, dict)
        lr_scheduler_cfg = opt_sched_cfg.get("lr_scheduler", None)
        assert isinstance(lr_scheduler_cfg, dict)
        scheduler = lr_scheduler_cfg.get("scheduler", None)
        assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
        assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)
        assert scheduler.start_factor == 0.2
        assert scheduler.end_factor == 0.8

    def test_test_step(
        self,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
    ) -> None:
        batch_size = n_history_steps = n_forecast_steps = 1
        batch = {
            cfg_input_space["name"]: torch.randn(
                batch_size,
                n_history_steps,
                cfg_input_space["channels"],
                cfg_input_space["shape"][0],
                cfg_input_space["shape"][1],
            ),
            cfg_output_space["name"]: torch.randn(
                batch_size,
                n_forecast_steps,
                cfg_output_space["channels"],
                cfg_output_space["shape"][0],
                cfg_output_space["shape"][1],
            ),
        }
        model = FakeDataModel(
            name="fake data",
            input_spaces=[cfg_input_space],
            n_forecast_steps=n_forecast_steps,
            n_history_steps=n_history_steps,
            output_space=cfg_output_space,
            optimizer=cfg_optimizer,
            scheduler=cfg_scheduler,
            loss=_DEFAULT_LOSS,
        )
        output_shape = batch["target"].shape
        output = model.test_step(batch, 0)
        assert isinstance(output, ModelTestOutput)
        assert output.prediction.shape == output_shape
        assert output.target.shape == output_shape
        assert output.loss.shape == torch.Size([])

    def test_training_step(
        self,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
    ) -> None:
        batch_size = n_history_steps = n_forecast_steps = 1
        batch = {
            cfg_input_space["name"]: torch.randn(
                batch_size,
                n_history_steps,
                cfg_input_space["channels"],
                cfg_input_space["shape"][0],
                cfg_input_space["shape"][1],
            ),
            cfg_output_space["name"]: torch.randn(
                batch_size,
                n_forecast_steps,
                cfg_output_space["channels"],
                cfg_output_space["shape"][0],
                cfg_output_space["shape"][1],
            ),
        }
        model = FakeDataModel(
            name="fake data",
            input_spaces=[cfg_input_space],
            n_forecast_steps=n_forecast_steps,
            n_history_steps=n_history_steps,
            output_space=cfg_output_space,
            optimizer=cfg_optimizer,
            scheduler=cfg_scheduler,
            loss=_DEFAULT_LOSS,
        )
        output = model.training_step(batch, 0)
        assert isinstance(output, torch.Tensor)
        assert output.shape == torch.Size([])

    def test_validation_step(
        self,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
    ) -> None:
        batch_size = n_history_steps = n_forecast_steps = 1
        batch = {
            cfg_input_space["name"]: torch.randn(
                batch_size,
                n_history_steps,
                cfg_input_space["channels"],
                cfg_input_space["shape"][0],
                cfg_input_space["shape"][1],
            ),
            cfg_output_space["name"]: torch.randn(
                batch_size,
                n_forecast_steps,
                cfg_output_space["channels"],
                cfg_output_space["shape"][0],
                cfg_output_space["shape"][1],
            ),
        }
        model = FakeDataModel(
            name="fake data",
            input_spaces=[cfg_input_space],
            n_forecast_steps=n_forecast_steps,
            n_history_steps=n_history_steps,
            output_space=cfg_output_space,
            optimizer=cfg_optimizer,
            scheduler=cfg_scheduler,
            loss=_DEFAULT_LOSS,
        )
        output = model.validation_step(batch, 0)
        assert isinstance(output, torch.Tensor)
        assert output.shape == torch.Size([])
