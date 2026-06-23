from typing import Any

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from icenet_mp.types import TensorNTCHW

from .base_model import BaseModel


class Persistence(BaseModel):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialise a Persistence model."""
        super().__init__(**kwargs)
        self.model = nn.Identity()
        self.automatic_optimization = False

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Persistence model does not need an optimizer."""
        return None

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        - start with multiple [NTCHW] inputs each with shape [batch, n_history_steps, C_input_k, H_input_k, W_input_k]
        - find the input with the same name as the output space
        - take the last time step and repeat it n_forecast_steps times
        """
        target_nchw = inputs[self.output_space.name][:, -1, :, :, :]
        return torch.stack([target_nchw for _ in range(self.n_forecast_steps)], dim=1)
