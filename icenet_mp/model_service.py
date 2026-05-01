import logging
from pathlib import Path, PosixPath
from typing import cast

import hydra
import torch
from lightning import Callback, Trainer, seed_everything
from lightning.fabric.utilities import suggested_max_num_workers
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib.runid import generate_id

from icenet_mp.callbacks import UnconditionalCheckpoint
from icenet_mp.data_loaders import CommonDataModule
from icenet_mp.models.base_model import BaseModel
from icenet_mp.types import SupportsMetadata
from icenet_mp.utils import get_device_name, get_timestamp, get_wandb_run

log = logging.getLogger(__name__)


class ModelService:
    def __init__(self, config: DictConfig) -> None:
        """Initialize the model service."""
        self.config_ = config
        if seed := config.get("seed", None):
            seed_everything(int(seed), workers=True)
        self.data_module_: CommonDataModule | None = None
        self.model_: BaseModel | None = None

    @classmethod
    def from_config(cls, config: DictConfig) -> "ModelService":
        """Build a new ModelService by loading a model from a configuration file."""
        # Load the model configuration
        builder = cls(config)

        # Construct the model
        log.info("Building a new %s model...", builder.config["model"]["name"])
        builder.model_ = hydra.utils.instantiate(
            dict(
                {
                    "hemisphere": builder.data_module.hemisphere,
                    "input_spaces": [
                        s.to_dict() for s in builder.data_module.input_spaces
                    ],
                    "latitudes": builder.data_module.latitudes,
                    "longitudes": builder.data_module.longitudes,
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
            log.debug("Found checkpoint at %s.", checkpoint_path)
        else:
            msg = f"Checkpoint file {checkpoint_path} does not exist."
            raise FileNotFoundError(msg)

        # Build a combined model configuration where the command line config takes
        # precedence except for the "model", "predict" and "train" keys which are
        # related to training the model.
        config_path = checkpoint_path.parent.parent / "files" / "model_config.yaml"
        try:
            # Load the model configuration from the checkpoint directory
            ckpt_config = DictConfig(OmegaConf.load(config_path))
            log.debug("Loaded checkpoint configuration from %s.", config_path)
            combined_cfg = DictConfig(OmegaConf.merge(ckpt_config, config))
            for key in ("model", "predict", "train"):
                combined_cfg[key] = OmegaConf.merge(
                    combined_cfg.get(key, {}), ckpt_config.get(key, {})
                )
        except (NotADirectoryError, FileNotFoundError):
            combined_cfg = config
            log.debug("Could not load checkpoint configuration from %s.", config_path)

        # Load the model from checkpoint
        builder = cls(combined_cfg)
        model_cls: type[BaseModel] = hydra.utils.get_class(
            builder.config["model"]["_target_"]
        )
        with torch.serialization.safe_globals([PosixPath]):
            log.info("Loading a trained %s model...", builder.config["model"]["name"])
            builder.model_ = model_cls.load_from_checkpoint(
                checkpoint_path,
                latitudes=builder.data_module.latitudes,
                longitudes=builder.data_module.longitudes,
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

    def build_run_directory(self, trainer: Trainer) -> Path:
        """Get run directory from Wandb or generate one in the same format."""
        # Get the run directory from Wandb if it exists
        wandb_run = get_wandb_run(trainer)
        if wandb_run:
            return Path(wandb_run._settings.sync_dir)

        # Otherwise generate a new run directory
        return (
            self.data_module.base_path
            / "training"
            / "local"
            / f"run-{get_timestamp()}-{generate_id()}"
        )

    def build_trainer(
        self,
        *,
        job_type: str,
    ) -> Trainer:
        """Configure the trainer with callbacks and loggers."""
        # Setup callbacks first
        callback_configs = self.config.get(job_type, {}).get("callbacks", {}).values()
        extra_callbacks = [
            hydra.utils.instantiate(callback_config)
            for callback_config in callback_configs
        ]
        if not extra_callbacks:
            log.warning("No callbacks have been set for the trainer.")

        # Setup lightning loggers
        logger_overrides = {
            "job_type": job_type,
            "project": job_type,
        }
        extra_loggers = [
            hydra.utils.instantiate(dict(**logger_config) | logger_overrides)
            for logger_config in self.config.get("loggers", {}).values()
        ]
        if not extra_loggers:
            log.warning("No loggers have been set for the trainer.")

        # Create a new trainer
        log.debug("Instantiating lightning trainer.")
        trainer = cast(
            "Trainer",
            hydra.utils.instantiate(
                dict(
                    {
                        "callbacks": extra_callbacks,
                        "deterministic": self.config.get("seed", None) is not None,
                        "logger": extra_loggers,
                    },
                    **self.config["train"]["trainer"],
                )
            ),
        )
        # Assign workers for data loading
        self.data_module.assign_workers(suggested_max_num_workers(trainer.num_devices))

        # Ensure the run directory exists
        run_directory = self.build_run_directory(trainer)
        log.debug("Set run directory to %s.", run_directory)
        run_directory.mkdir(parents=True, exist_ok=True)

        # Save model config to the run directory
        model_config_path = run_directory / "files" / "model_config.yaml"
        if trainer.is_global_zero:
            model_config_path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(self.config, model_config_path)
            if wandb_run := get_wandb_run(trainer):
                wandb_run.save(model_config_path, base_path=model_config_path.parent)

        # Additional configuration for callbacks
        for callback in cast("list[Callback]", trainer.callbacks):  # type: ignore[attr-defined]
            log.debug("Configuring callback %s.", callback.__class__.__name__)
            # Set metadata for supported callbacks
            if isinstance(callback, SupportsMetadata):
                log.debug("Setting metadata for %s.", callback.__class__.__name__)
                callback.set_metadata(self.config, self.model.__class__.__name__)
            # Set checkpoint run directory for supported callbacks
            if isinstance(callback, (ModelCheckpoint, UnconditionalCheckpoint)):
                log.debug(
                    "Setting run_directory for %s to %s.",
                    callback.__class__.__name__,
                    run_directory / "checkpoints",
                )
                callback.dirpath = run_directory / "checkpoints"

        return trainer

    def evaluate(self) -> None:
        """Evaluate a trained model."""
        # Configure the trainer with evaluation callbacks and loggers
        log.info("Configuring model for evaluation.")
        trainer = self.build_trainer(job_type="evaluate")
        # Log evaluation details
        log.info(
            "Starting evaluation using %d threads across %d %s device(s).",
            torch.get_num_threads(),
            trainer.num_devices,
            get_device_name(trainer.accelerator.name()),
        )

        # Evaluate the model
        trainer.test(
            model=self.model,
            datamodule=self.data_module,
        )

    def train(self) -> None:
        """Train a model."""
        # Configure the trainer with training callbacks and loggers
        log.info("Configuring model for training.")
        trainer = self.build_trainer(job_type="train")

        # Log training details
        log.info(
            "Starting training for %d epochs using %d threads across %d %s device(s).",
            trainer.max_epochs,
            torch.get_num_threads(),
            trainer.num_devices,
            get_device_name(trainer.accelerator.name()),
        )

        # Train the model
        trainer.fit(
            model=self.model,
            datamodule=self.data_module,
        )
