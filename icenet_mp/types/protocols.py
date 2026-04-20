from typing import Protocol, runtime_checkable

from omegaconf import DictConfig


@runtime_checkable
class SupportsLatLon(Protocol):
    def set_latlon(
        self, name: str, latitudes: list[float], longitudes: list[float]
    ) -> None: ...


@runtime_checkable
class SupportsMetadata(Protocol):
    def set_metadata(self, config: DictConfig, model_name: str) -> None: ...
