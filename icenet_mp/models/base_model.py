import itertools
from abc import ABC, abstractmethod
from typing import Any

import hydra
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerConfig,
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
)
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from icenet_mp.models.metrics.base_metrics import MAEDaily, RMSEDaily
from icenet_mp.models.metrics.sie_error_abs import SIEErrorDaily
from icenet_mp.types import DataSpace, ModelTestOutput, TensorNTCHW


class BaseModel(LightningModule, ABC):
    """A base class for all models used in the IceNet-MP project."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        name: str,
        input_spaces: list[DictConfig],
        n_forecast_steps: int,
        n_history_steps: int,
        output_space: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        **_kwargs: Any,
    ) -> None:
        """Initialise a BaseModel.

        Input spaces and the desired output space must be specified, as must the number
        of forecast and history steps.

        Optimizer configuration is also set here.
        """
        super().__init__()

        # Save model name
        self.name = name

        # Save history and forecast steps
        if n_forecast_steps <= 0:
            msg = "Number of forecast steps must be greater than 0."
            raise ValueError(msg)
        self.n_forecast_steps = n_forecast_steps
        if n_history_steps <= 0:
            msg = "Number of history steps must be greater than 0."
            raise ValueError(msg)
        self.n_history_steps = n_history_steps

        # Construct the input and output spaces
        self.input_spaces = [DataSpace.from_dict(space) for space in input_spaces]
        self.output_space = DataSpace.from_dict(output_space)

        # Store the optimizer and scheduler configs
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

        self.test_metrics = MetricCollection(
            {"sieerror": SIEErrorDaily(), "rmse": RMSEDaily(), "mae": MAEDaily()}
        )

        # Save all of the arguments to __init__ as hyperparameters
        # This will also save the parameters of whichever child class is used
        # Note that W&B will log all hyperparameters
        self.save_hyperparameters()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Construct the optimizer and optional scheduler from the config."""
        # Optimizer
        optimizer = hydra.utils.instantiate(
            dict(**self.optimizer_cfg)
            | {
                "params": itertools.chain(
                    *[module.parameters() for module in self.children()]
                )
            }
        )
        # If no scheduler config is provided, return just the optimizer
        if not self.scheduler_cfg:
            return OptimizerConfig(optimizer=optimizer)

        # Scheduler
        scheduler_args = self.scheduler_cfg
        scheduler = hydra.utils.instantiate(
            {
                "_target_": scheduler_args.pop("_target_"),
                "optimizer": optimizer,
                **scheduler_args.pop("scheduler_parameters", {}),
            }
        )

        # Return the optimizer and scheduler
        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler=LRSchedulerConfigType(scheduler=scheduler, **scheduler_args),
        )

    @abstractmethod
    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        - start with multiple [NTCHW] inputs, one for each input dataset
        - return a single [NTCHW] output representing the predicted output

        Args:
            inputs: Dictionary of dataset name to TensorNTCHW with shape [batch, n_history_steps, C_input_k, H_input_k, W_input_k]

        Returns:
            Predicted TensorNTCHW with shape [batch, n_forecast_steps, C_output, H_output, W_output]

        """

    def loss(self, prediction: TensorNTCHW, target: TensorNTCHW) -> torch.Tensor:
        """Calculate the loss given a prediction and target."""
        return torch.nn.functional.l1_loss(prediction, target)

    def test_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,  # noqa: PT019
    ) -> ModelTestOutput:
        """Run the test step, in PyTorch eval model (i.e. no gradients).

        - Separate the batch into inputs and target
        - Run inputs through the model
        - Return the prediction, target and loss

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A ModelTestOutput containing the prediction, target and loss for the batch.

        """
        target = batch.pop("target")
        prediction = self(batch)
        loss = self.loss(prediction, target)
        # update test metrics with the current batch; computation will be done at epoch end
        self.test_metrics.update(prediction, target)

        return ModelTestOutput(prediction, target, loss)

    def training_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> torch.Tensor:
        """Run the training step.

        - Separate the batch into inputs and target
        - Run inputs and target through the model
        - Calculate the loss wrt. the target

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A Tensor containing the loss for the batch.

        """
        target = batch["target"].clone().detach()
        prediction = self(batch)
        loss = self.loss(prediction, target)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> torch.Tensor:
        """Run the validation step.

        A batch contains one tensor for each input dataset and one for the target
        These are [NTCHW] tensors with (batch_size, n_history_steps, C, H, W)

        - Separate the batch into inputs and target
        - Run inputs through the model
        - Calculate and log the loss wrt. the target

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A Tensor containing the loss for the batch.

        """
        target = batch["target"].clone().detach()
        prediction = self(batch)
        loss = self.loss(prediction, target)
        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
