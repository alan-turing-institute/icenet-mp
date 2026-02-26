from datetime import UTC, datetime

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from wandb.wandb_run import Run


def datetime_from_npdatetime(dt: np.datetime64) -> datetime:
    """Convert numpy datetime64 to aware datetime in UTC."""
    return dt.astype("datetime64[ms]").astype(datetime).astimezone(UTC)


def get_device_name(accelerator_name: str) -> str:
    """Get the device name for the given accelerator."""
    if accelerator_name == "cuda":
        try:
            return torch.cuda.get_device_name()
        except AssertionError:
            return "Unknown CUDA device"
    if accelerator_name == "mps":
        return "Apple Silicon GPU"
    if accelerator_name == "xpu":
        try:
            return torch.xpu.get_device_name()
        except AssertionError:
            return "Unknown XPU device"
    return "CPU"


def get_timestamp() -> str:
    """Return the current time as a string."""
    return datetime.now(tz=UTC).strftime(r"%Y%m%d_%H%M%S")


def get_wandb_run(trainer: Trainer) -> Run | None:
    """Get the Wandb Run instance if it exists."""
    for lightning_logger in trainer.loggers:
        if isinstance(lightning_logger, WandbLogger) and isinstance(
            experiment := lightning_logger.experiment, Run
        ):
            return experiment
    return None


def normalise_date(np_datetime: np.datetime64) -> np.datetime64:
    """Normalise a datetime to midnight."""
    dt: datetime = np_datetime.astype("datetime64[ms]").astype(datetime)
    return np.datetime64(dt.date())


def to_list(value: str | list[str]) -> list[str]:
    """Convert a string or list of strings to a list of strings."""
    if isinstance(value, str):
        return [value]
    return value
