import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from icenet_mp.losses.rmse_loss import RMSELoss
from tests.models.test_base_model import FakeDataModel

LOSS_CONFIGS = {
    "mse": OmegaConf.create({"_target_": "torch.nn.MSELoss"}),
    "mae": OmegaConf.create({"_target_": "torch.nn.L1Loss"}),
    "huber": OmegaConf.create({"_target_": "torch.nn.HuberLoss", "delta": 0.5}),
    "smooth_l1": OmegaConf.create({"_target_": "torch.nn.SmoothL1Loss", "beta": 0.5}),
    "rmse": OmegaConf.create({"_target_": "icenet_mp.losses.rmse_loss.RMSELoss"}),
}

LOSS_TYPES = {
    "mse": torch.nn.MSELoss,
    "mae": torch.nn.L1Loss,
    "huber": torch.nn.HuberLoss,
    "smooth_l1": torch.nn.SmoothL1Loss,
    "rmse": RMSELoss,
}


class TestLossConfig:
    """Tests that the loss function config is correctly picked up by BaseModel."""

    # 1. Model cannot be initialised without a loss section
    def test_missing_loss_raises(
        self, cfg_input_space: DictConfig, cfg_output_space: DictConfig
    ) -> None:
        with pytest.raises(TypeError):
            FakeDataModel(
                name="fake data",
                input_spaces=[cfg_input_space],
                n_forecast_steps=1,
                n_history_steps=1,
                output_space=cfg_output_space,
                optimizer=DictConfig({}),
                scheduler=DictConfig({}),
                # loss intentionally omitted
            )
