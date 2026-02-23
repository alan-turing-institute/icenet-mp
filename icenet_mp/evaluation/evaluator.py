import logging
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING

import hydra

# Set matplotlib backend BEFORE any imports that might use it
import torch
from lightning.fabric.utilities import suggested_max_num_workers
from omegaconf import DictConfig, OmegaConf

from icenet_mp.data_loaders import CommonDataModule
from icenet_mp.utils import get_device_name, get_device_threads, get_timestamp

if TYPE_CHECKING:
    from lightning import Callback, Trainer

    from icenet_mp.models import BaseModel

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """A wrapper for PyTorch evaluation."""

    def __init__(self, config: DictConfig, checkpoint_path: Path) -> None:
        """Initialize the model evaluator from a config and checkpoint."""
        # Verify the checkpoint path
        if checkpoint_path.exists():
            logger.debug("Loaded checkpoint from %s.", checkpoint_path)
        else:
            msg = f"Checkpoint file {checkpoint_path} does not exist."
            raise FileNotFoundError(msg)

        # Load the model configuration
        config_path = checkpoint_path.parent.parent / "model_config.yaml"
        try:
            ckpt_config = OmegaConf.load(config_path)
            logger.debug("Loaded checkpoint config from %s.", ckpt_config)
            config["model"]["_target_"] = ckpt_config["model"]["_target_"]  # type: ignore[index]
        except (NotADirectoryError, FileNotFoundError):
            logger.debug("Could not find model configuration file at %s.", config_path)

        # Load the model from checkpoint
        model_cls: type[BaseModel] = hydra.utils.get_class(config["model"]["_target_"])
        with torch.serialization.safe_globals([PosixPath]):
            self.model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint_path)

        # Load inputs into a data module
        self.data_module = CommonDataModule(config)

        # Add callbacks
        callbacks: list[Callback] = []
        for cfg in config.get("evaluate", {}).get("callbacks", {}).values():
            callback = hydra.utils.instantiate(cfg)
            logger.debug("Adding evaluation callback %s.", callback.__class__.__name__)
            if hasattr(callback, "set_metadata"):
                callback.set_metadata(config, model_cls.__name__)
            callbacks.append(callback)

        # Construct lightning loggers
        lightning_loggers = [
            hydra.utils.instantiate(
                dict(**logger_config)
                | {
                    "job_type": "evaluate",
                    "name": f"{self.model.name}-{get_timestamp()}",
                    "project": "leaderboard",
                },
            )
            for logger_config in config.get("loggers", {}).values()
        ]

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

        # Assign workers for data loading
        self.data_module.assign_workers(
            suggested_max_num_workers(self.trainer.num_devices)
        )

    def evaluate(self) -> None:
        logger.info(
            "Starting evaluation using %d threads across %d %s device(s).",
            get_device_threads(),
            self.trainer.num_devices,
            get_device_name(self.trainer.accelerator.__class__.__name__), # name() caused CUDA acceleator error
        )
        self.trainer.test(
            model=self.model,
            datamodule=self.data_module,
        )
