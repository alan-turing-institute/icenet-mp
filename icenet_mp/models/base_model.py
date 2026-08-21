from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from functools import cached_property, partial
from pathlib import Path
from typing import Any, ClassVar

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
from torchmetrics import Metric, MetricCollection

from icenet_mp.metrics import (
    CentroidErrorPerForecastDay,
    DistanceAveragedIceEdgeErrorPerForecastDay,
    FractionalSkillScorePerForecastDay,
    IceNetAccuracyPerForecastDay,
    IntegratedIceEdgeErrorPerForecastDay,
    MAEPerForecastDay,
    RMSEPerForecastDay,
    SeaIceExtentErrorPerForecastDay,
    SSIMPerForecastDay,
)
from icenet_mp.models.common import Mask
from icenet_mp.types import (
    DataSpace,
    Hemisphere,
    MaskType,
    ModelStepOutput,
    TensorNTCHW,
)


class BaseModel(LightningModule, ABC):
    """A base class for all models used in the IceNet-MP project."""

    # Parameters that should be excluded from hyperparameter logging
    ignored_hparams: ClassVar[frozenset[str]] = frozenset(
        ("latitudes_fn", "longitudes_fn", "mask_dir")
    )

    def __init__(  # noqa: PLR0913
        self,
        *,
        hemisphere: Hemisphere,
        input_spaces: list[DictConfig],
        latitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        longitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        loss: DictConfig,
        mask_dir: str | Path | None = None,
        metrics: list[str] | None = None,
        n_forecast_steps: int,
        n_history_steps: int,
        name: str,
        optimizer: DictConfig,
        output_space: DictConfig,
        scheduler: DictConfig,
        **_kwargs: Any,
    ) -> None:
        """Initialise a BaseModel.

        Input spaces and the desired output space must be specified, as must the number
        of forecast and history steps.

        Optimizer configuration is also set here.

        ``mask_dir``, if given, is a directory holding `land_mask.npy` (generated for
        SSMIS datasets by `datasets create`). When present, the ``"diiee"``/``"fss_*"``
        metrics use it to exclude land/ice boundaries from ice-edge detection, so only
        ocean ice/no-ice transitions count as the sea-ice edge.

        The ``metrics`` parameter controls which metrics are computed during training,
        validation, and testing. Defaults to ``["accuracy", "mae", "rmse", "sieerror",
        "iiee", "diiee", "centroid_error", "fss_1", "fss_5", "fss_15", "ssim"]``.
        ``"iiee"`` is the Integrated Ice Edge Error, the area of disagreement between
        predicted and true ice extent (see ``icenet_mp.metrics.iiee``); unlike
        ``"sieerror"`` it never lets over- and under-estimation cancel out.
        ``"diiee"`` is the Distance-averaged IIEE: the IIEE area normalised by
        combined ice-edge length, giving an average edge displacement in km rather
        than an area (see ``icenet_mp.metrics.diiee``); it is undefined (NaN) for
        lead times where neither field has any ice edge. ``"centroid_error"`` is
        the value-weighted centre-of-mass distance (only meaningful for synthetic checks
        where the field is a single blob); ``"fss_1"``/``"fss_5"``/``"fss_15"`` are the
        Fractional Skill Score of the sea-ice edge at neighbourhood sizes of 1, 5, and
        15 pixels respectively (see ``icenet_mp.metrics.fss``); ``"ssim"`` is the
        Structural Similarity Index (see ``icenet_mp.metrics.ssim``).
        """
        super().__init__()

        # Save model name, hemisphere and lat/lon information
        self.name = name
        self.hemisphere: Hemisphere = hemisphere
        self.latitudes_fn = latitudes_fn
        self.longitudes_fn = longitudes_fn

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

        # Store the optimizer, scheduler and loss configs
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.loss_cfg = loss

        # Land mask for ice-edge metrics (excludes land/ice boundaries from FSS/DIIEE)
        land_mask: torch.Tensor | None = None
        if mask_dir is not None:
            land_mask = Mask(
                mask_type=MaskType.LAND,
                output_shape=self.output_space.shape,
                mask_dir=mask_dir,
            ).mask.bool()

        # Metrics
        _metric_classes: dict[str, Callable[[], Metric]] = {
            "accuracy": partial(IceNetAccuracyPerForecastDay, land_mask=land_mask),
            "mae": partial(MAEPerForecastDay, land_mask=land_mask),
            "rmse": partial(RMSEPerForecastDay, land_mask=land_mask),
            "sieerror": partial(SeaIceExtentErrorPerForecastDay, land_mask=land_mask),
            "iiee": partial(IntegratedIceEdgeErrorPerForecastDay, land_mask=land_mask),
            "diiee": partial(
                DistanceAveragedIceEdgeErrorPerForecastDay, land_mask=land_mask
            ),
            "centroid_error": partial(CentroidErrorPerForecastDay, land_mask=land_mask),
            "fss_1": partial(
                FractionalSkillScorePerForecastDay,
                neighborhood_size=1,
                land_mask=land_mask,
            ),
            "fss_5": partial(
                FractionalSkillScorePerForecastDay,
                neighborhood_size=5,
                land_mask=land_mask,
            ),
            "fss_15": partial(
                FractionalSkillScorePerForecastDay,
                neighborhood_size=15,
                land_mask=land_mask,
            ),
            "ssim": partial(SSIMPerForecastDay, land_mask=land_mask),
        }
        metric_names = (
            metrics
            if metrics is not None
            else [
                "accuracy",
                "mae",
                "rmse",
                "sieerror",
                "iiee",
                "diiee",
                "centroid_error",
                "fss_1",
                "fss_5",
                "fss_15",
                "ssim",
            ]
        )
        _common_metrics: dict[str, Metric | MetricCollection] = {
            name: _metric_classes[name]() for name in metric_names
        }
        self.test_metrics = MetricCollection(deepcopy(_common_metrics))
        self.train_metrics = MetricCollection(deepcopy(_common_metrics))
        self.validation_metrics = MetricCollection(deepcopy(_common_metrics))

        # All arguments to the ultimate child class will be logged as hyperparameters,
        # and saved to W&B, unless explicitly ignored here.
        self.save_hyperparameters(ignore=[*self.ignored_hparams])

    @cached_property
    def latitudes(self) -> dict[str, list[float]]:
        return {} if not self.latitudes_fn else self.latitudes_fn()

    @cached_property
    def longitudes(self) -> dict[str, list[float]]:
        return {} if not self.longitudes_fn else self.longitudes_fn()

    @property
    def multistage_only(self) -> bool:
        return False

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Construct the optimizer and optional scheduler from the config."""
        # Optimizer
        optimizer = hydra.utils.instantiate(
            self.optimizer_cfg,
            params=filter(lambda p: p.requires_grad, self.parameters()),
        )

        # If no scheduler config is provided, return just the optimizer
        if not self.scheduler_cfg:
            return OptimizerConfig(optimizer=optimizer)

        # Scheduler
        scheduler = hydra.utils.instantiate(
            self.scheduler_cfg["scheduler_parameters"],
            _target_=self.scheduler_cfg["_target_"],
            optimizer=optimizer,
        )

        # Return the optimizer and scheduler
        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler=LRSchedulerConfigType(
                scheduler=scheduler, **self.scheduler_cfg["lr_scheduler_parameters"]
            ),
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

    @property
    def loss_cfg(self) -> DictConfig:
        """Get the loss configuration."""
        return self._loss_cfg

    @loss_cfg.setter
    def loss_cfg(self, cfg: DictConfig) -> None:
        """Set the loss configuration and instantiate the loss function."""
        self.loss_fn = hydra.utils.instantiate(cfg)
        if not isinstance(self.loss_fn, torch.nn.Module):
            msg = (
                f"Loss `_target_` {cfg.get('_target_', '(missing)')!r} created a "
                f"{type(self.loss_fn).__name__}, expected a torch.nn.Module."
            )
            raise TypeError(msg)
        self._loss_cfg = cfg

    def loss(self, prediction: TensorNTCHW, target: TensorNTCHW) -> torch.Tensor:
        """Calculate the loss given a prediction and target."""
        return self.loss_fn(prediction, target)

    def process_batch(self, batch: dict[str, TensorNTCHW]) -> dict[str, TensorNTCHW]:
        """Process a batch before the forward pass and loss computation.

        Subclasses can override this to extract or transform inputs before the standard
        training/validation steps. The returned dict must include a ``"target"`` key.
        """
        return batch

    def test_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,  # noqa: PT019
    ) -> ModelStepOutput:
        """Run the test step, in PyTorch eval model (i.e. no gradients).

        - Separate the batch into inputs and target
        - Run inputs through the model
        - Return the prediction, target and loss

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A ModelStepOutput containing the prediction, target and loss for the batch.

        """
        batch = self.process_batch(batch)
        target = batch.pop("target")
        prediction = self(batch)
        loss = self.loss(prediction, target)

        # Log metrics; computation will be done at epoch end
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.test_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)

    def training_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> ModelStepOutput:
        """Run the training step.

        - Separate the batch into inputs and target
        - Run inputs and target through the model
        - Calculate the loss wrt. the target

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A ModelStepOutput containing the prediction, target and loss for the batch.

        """
        batch = self.process_batch(batch)
        target = batch["target"].clone().detach()
        prediction = self(batch)
        loss = self.loss(prediction, target)

        # Log metrics; computation will be done at epoch end
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)

    def validation_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> ModelStepOutput:
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
            A ModelStepOutput containing the prediction, target and loss for the batch.

        """
        batch = self.process_batch(batch)
        target = batch["target"].clone().detach()
        prediction = self(batch)
        loss = self.loss(prediction, target)

        # Log metrics; computation will be done at epoch end
        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.validation_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)
