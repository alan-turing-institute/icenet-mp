import logging
from collections.abc import Iterable
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, cast

import hydra
import torch
from lightning import Callback, Trainer
from lightning.fabric.utilities import suggested_max_num_workers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib.runid import generate_id
from wandb.wandb_run import Run

from icenet_mp.callbacks import UnconditionalCheckpoint
from icenet_mp.data_loaders import CommonDataModule
from icenet_mp.models.base_model import BaseModel
from icenet_mp.types import SupportsMetadata
from icenet_mp.utils import get_device_name, get_timestamp

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger as LightningLogger

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, config: DictConfig) -> None:
        """Initialize the model service."""
        self.config_ = config
        self.data_module_: CommonDataModule | None = None
        self.model_: BaseModel | None = None
        self.trainer_: Trainer | None = None
        self.extra_callbacks_: list[Callback] = []
        self.extra_loggers_: list[LightningLogger] = []
        self.run_directory_: Path | None = None

    @classmethod
    def from_config(cls, config: DictConfig) -> "ModelService":
        """Build a new ModelService by loading a model from a configuration file."""
        # Load the model configuration
        builder = cls(config)

        # Construct the model
        builder.model_ = hydra.utils.instantiate(
            dict(
                {
                    "input_spaces": [
                        s.to_dict() for s in builder.data_module.input_spaces
                    ],
                    "n_forecast_steps": builder.data_module.n_forecast_steps,
                    "n_history_steps": builder.data_module.n_history_steps,
                    "output_space": builder.data_module.output_space.to_dict(),
                    "optimizer": config["train"]["optimizer"],
                    "scheduler": config["train"]["scheduler"],
                },
                **config["model"],
            ),
            _recursive_=False,
            _convert_="object",
        )

        # Return the builder
        return builder

    @classmethod
    def from_checkpoint(
        cls, config: DictConfig, checkpoint_path: Path
    ) -> "ModelService":
        """Build a new ModelService by loading a model from a checkpoint."""
        # Verify the checkpoint path
        if checkpoint_path.is_file():
            logger.debug("Found checkpoint at %s.", checkpoint_path)
        else:
            msg = f"Checkpoint file {checkpoint_path} does not exist."
            raise FileNotFoundError(msg)

        # Build a combined model configuration where the command line config takes
        # precedence except for the "model", "predict" and "train" keys which are
        # related to training the model.
        config_path = checkpoint_path.parent.parent / "model_config.yaml"
        try:
            # Load the model configuration from the checkpoint directory
            ckpt_config = DictConfig(OmegaConf.load(config_path))
            logger.debug("Loaded checkpoint configuration from %s.", config_path)
            combined_cfg = DictConfig(OmegaConf.merge(ckpt_config, config))
            for key in ("model", "predict", "train"):
                combined_cfg[key] = OmegaConf.merge(
                    combined_cfg.get(key, {}), ckpt_config.get(key, {})
                )
        except (NotADirectoryError, FileNotFoundError):
            combined_cfg = config
            logger.debug(
                "Could not load checkpoint configuration from %s.", config_path
            )

        # Load the model from checkpoint
        builder = cls(combined_cfg)
        model_cls: type[BaseModel] = hydra.utils.get_class(
            builder.config["model"]["_target_"]
        )
        with torch.serialization.safe_globals([PosixPath]):
            builder.model_ = model_cls.load_from_checkpoint(
                checkpoint_path=checkpoint_path
            )

        return builder

    @property
    def config(self) -> DictConfig:
        """Get the model configuration."""
        if not self.config_:
            msg = "Model config has not been initialised."
            raise AttributeError(msg)
        return self.config_

    @property
    def data_module(self) -> CommonDataModule:
        """Get the data module instance."""
        if not self.data_module_:
            self.data_module_ = CommonDataModule(self.config)
        return self.data_module_

    @property
    def model(self) -> BaseModel:
        """Get the model instance."""
        if not self.model_:
            msg = "Model has not been initialised."
            raise AttributeError(msg)
        return self.model_

    @property
    def run_directory(self) -> Path:
        """Get run directory from wandb logger or generate one in the same format."""
        if not self.run_directory_:
            # Get the run directory from the WandbLogger if it exists
            for lightning_logger in self.trainer.loggers:
                if not isinstance(lightning_logger, WandbLogger):
                    continue
                if not isinstance(experiment := lightning_logger.experiment, Run):
                    continue
                self.run_directory_ = Path(experiment._settings.sync_dir)
                break

            # Otherwise generate a new run directory
            if not self.run_directory_:
                self.run_directory_ = (
                    self.data_module.base_path
                    / "training"
                    / "local"
                    / f"run-{get_timestamp()}-{generate_id()}"
                )

            # Ensure the run directory exists
            logger.debug("Set run directory to %s.", self.run_directory_)
            self.run_directory_.mkdir(parents=True, exist_ok=True)
        return self.run_directory_

    @property
    def trainer(self) -> Trainer:
        """Create a new Trainer or return the existing one."""
        if not self.trainer_:
            # Create a new Trainer
            logger.debug("Instantiating lightning trainer.")
            self.trainer_ = cast(
                "Trainer",
                hydra.utils.instantiate(
                    dict(
                        {
                            "callbacks": self.extra_callbacks_,
                            "logger": self.extra_loggers_,
                        },
                        **self.config["train"]["trainer"],
                    )
                ),
            )
            # Assign workers for data loading
            self.data_module.assign_workers(
                suggested_max_num_workers(self.trainer_.num_devices)
            )
        return self.trainer_

    def add_callbacks(self, callback_configs: Iterable[DictConfig]) -> None:
        """Add extra lightning callbacks."""
        self.extra_callbacks_ += [
            hydra.utils.instantiate(callback_config)
            for callback_config in callback_configs
        ]

    def add_loggers(self, overrides: dict[str, str]) -> None:
        """Add extra lightning loggers."""
        self.extra_loggers_ += [
            hydra.utils.instantiate(dict(**logger_config) | overrides)
            for logger_config in self.config.get("loggers", {}).values()
        ]

    def configure_trainer(
        self,
        *,
        job_type: str,
    ) -> None:
        """Configure the trainer with callbacks and loggers."""
        # Setup callbacks first
        callback_configs = self.config.get(job_type, {}).get("callbacks", {}).values()
        self.add_callbacks(callback_configs)
        if not self.extra_callbacks_:
            logger.warning("No callbacks have been set for the trainer.")

        # Setup lightning loggers
        logger_overrides = {
            "job_type": job_type,
            "project": job_type,
        }
        self.add_loggers(logger_overrides)
        if not self.extra_loggers_:
            logger.warning("No loggers have been set for the trainer.")

        # Additional configuration for callbacks
        for callback in cast("list[Callback]", self.trainer.callbacks):  # type: ignore[attr-defined]
            logger.debug("Configuring callback %s.", callback.__class__.__name__)
            # Set metadata for supported callbacks
            if isinstance(callback, SupportsMetadata):
                logger.debug("Setting metadata for %s.", callback.__class__.__name__)
                callback.set_metadata(self.config, self.model.__class__.__name__)
            # Set checkpoint run directory for supported callbacks
            if isinstance(callback, (ModelCheckpoint, UnconditionalCheckpoint)):
                logger.debug(
                    "Setting run_directory for %s to %s.",
                    callback.__class__.__name__,
                    self.run_directory / "checkpoints",
                )
                callback.dirpath = self.run_directory / "checkpoints"

        # Save model config to the run directory
        OmegaConf.save(self.config, self.run_directory / "model_config.yaml")

    def evaluate(self) -> None:
        """Evaluate a trained model."""
        # Configure the trainer with evaluation callbacks and loggers
        logger.info("Configuring model for evaluation.")
        self.configure_trainer(job_type="evaluate")
        # Log evaluation details
        logger.info(
            "Starting evaluation using %d threads across %d %s device(s).",
            torch.get_num_threads(),
            self.trainer.num_devices,
            get_device_name(self.trainer.accelerator.name()),
        )

        # Evaluate the model
        self.trainer.test(
            model=self.model,
            datamodule=self.data_module,
        )

    def train(self) -> None:
        """Train a model."""
        # Configure the trainer with training callbacks and loggers
        logger.info("Configuring model for training.")
        self.configure_trainer(job_type="train")

        # Log training details
        logger.info(
            "Starting training for %d epochs using %d threads across %d %s device(s).",
            self.trainer.max_epochs,
            torch.get_num_threads(),
            self.trainer.num_devices,
            get_device_name(self.trainer.accelerator.name()),
        )

        # Train the model
        self.trainer.fit(
            model=self.model,
            datamodule=self.data_module,
        )
