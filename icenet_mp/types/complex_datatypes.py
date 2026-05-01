from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from omegaconf import DictConfig
from torch import Tensor

from .typedefs import TensorNTCHW


class DataSpace:
    """Description of a CHW data space."""

    channels: int
    name: str
    shape: tuple[int, int]

    def __init__(self, channels: int, name: str, shape: Sequence[int]) -> None:
        """Initialise a DataSpace from channels, name and shape."""
        self.channels = int(channels)
        self.name = name
        self.shape = (int(shape[0]), int(shape[1]))

    @property
    def area(self) -> int:
        """Return the area of the data space."""
        return self.shape[0] * self.shape[1]

    @property
    def chw(self) -> tuple[int, int, int]:
        """Return a tuple of [channels, height, width]."""
        return (self.channels, *self.shape)

    @classmethod
    def from_dict(cls, config: DictConfig | dict[str, Any]) -> Self:
        return cls(
            channels=config["channels"], name=config["name"], shape=config["shape"]
        )

    def to_dict(self) -> DictConfig:
        """Return the DataSpace as a DictConfig."""
        return DictConfig(
            {"channels": self.channels, "name": self.name, "shape": self.shape}
        )


@dataclass
class ModelTestOutput(Mapping[str, Tensor]):
    """Output of a model test step."""

    prediction: TensorNTCHW
    target: TensorNTCHW
    loss: Tensor

    def __getitem__(self, key: str) -> Tensor:
        """Get a tensor by key."""
        if key == "prediction":
            return self.prediction
        if key == "target":
            return self.target
        if key == "loss":
            return self.loss
        msg = f"Key {key} not found in ModelTestOutput"
        raise KeyError(msg)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys of ModelTestOutput."""
        yield "prediction"
        yield "target"
        yield "loss"

    def __len__(self) -> int:
        """Return ModelTestOutput length."""
        return 3
