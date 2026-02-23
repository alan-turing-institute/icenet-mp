import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from lightning.fabric.utilities import suggested_max_num_workers
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from icenet_mp.callbacks import UnconditionalCheckpoint
from icenet_mp.data_loaders import CommonDataModule
from icenet_mp.utils import (
    generate_run_name,
    get_device_name,
    get_device_threads,
    get_wandb_logger,
)

if TYPE_CHECKING:
    from lightning import Callback, Trainer

    from icenet_mp.models import BaseModel


logger = logging.getLogger(__name__)


class ModelTrainer:
    """A wrapper for PyTorch training."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the model trainer."""
        # Load inputs into a data module
        self.data_module = CommonDataModule(config)

        # Construct the model
        self.model: BaseModel = hydra.utils.instantiate(
            dict(
                {
                    "input_spaces": [
                        s.to_dict() for s in self.data_module.input_spaces
                    ],
                    "n_forecast_steps": self.data_module.n_forecast_steps,
                    "n_history_steps": self.data_module.n_history_steps,
                    "output_space": self.data_module.output_space.to_dict(),
                    "optimizer": config["train"]["optimizer"],
                    "scheduler": config["train"]["scheduler"],
                },
                **config["model"],
            ),
            _recursive_=False,
            _convert_="object",
        )

        # Construct lightning loggers
        lightning_loggers = [
            hydra.utils.instantiate(
                dict(
                    {
                        "job_type": "train",
                        "project": self.model.name,
                    },
                    **logger_config,
                )
            )
            for logger_config in config.get("loggers", {}).values()
        ]

        # Get run directory from wandb logger or generate a new one
        if wandb_logger := get_wandb_logger(lightning_loggers):
            # run_directory = Path(wandb_logger.experiment._settings.sync_dir) this was causing AttributeError: 'function' object has no attribute 'sync_dir' error with ddp
            # run_directory = Path(wandb_logger.save_dir) / "wandb" / generate_run_name() # worked but wrong checkpoint location
            # _ = wandb_logger.experiment  # Try to initialize
            try:           
                run_directory = Path(wandb_logger.experiment.dir).parent
                logger.warning(f'Using W&B dir: {run_directory}')
            except (AttributeError, TypeError):
                run_directory = Path("/tmp/unused-rank-fallback") 
                logger.warning(f'Using fallback dir: {run_directory}')
            run_directory.mkdir(parents=True, exist_ok=True)
            
        else:
            run_directory = (
                self.data_module.base_path / "training" / "local" / generate_run_name()
            )
            run_directory.mkdir(parents=True, exist_ok=True)

        # Add callbacks
        callbacks: list[Callback] = [
            hydra.utils.instantiate(cfg)
            for cfg in config["train"].get("callbacks", {}).values()
        ]
        for callback in callbacks:
            logger.debug("Adding training callback %s.", callback.__class__.__name__)

        # Construct the trainer
        self.trainer: Trainer = hydra.utils.instantiate(
            dict(
                {
                    "callbacks": callbacks,
                    "logger": lightning_loggers,
                },
                **config["train"]["trainer"],
            )
        )

        # Set properties for checkpoint callbacks
        for callback in self.trainer.callbacks:
            if isinstance(callback, (ModelCheckpoint, UnconditionalCheckpoint)):
                callback.dirpath = run_directory / "checkpoints"

        # Assign workers for data loading
        self.data_module.assign_workers(
            suggested_max_num_workers(self.trainer.num_devices)
        )

        # Save config to the output directory
        OmegaConf.save(config, run_directory / "model_config.yaml")

    def train(self) -> None:
        logger.info(
            "Starting training for %d epochs using %d threads across %d %s device(s).",
            self.trainer.max_epochs,
            get_device_threads(),
            self.trainer.num_devices,
            get_device_name(self.trainer.accelerator.__class__.__name__), # .name caused CUDA accelerator version mismatch
        )
        self.trainer.fit(
            model=self.model,
            datamodule=self.data_module,
        )
