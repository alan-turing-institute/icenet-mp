from typing import Protocol, runtime_checkable

from omegaconf import DictConfig


@runtime_checkable
class SupportsMetadata(Protocol):
    def set_metadata(self, config: DictConfig, model_name: str) -> None: ...
