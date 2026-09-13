from typing import Any, Protocol, runtime_checkable

from omegaconf import DictConfig


@runtime_checkable
class SupportsImageLogging(Protocol):
    def log_image(
        self, key: str, images: list[Any], step: int | None = None, **kwargs: Any
    ) -> None: ...


@runtime_checkable
class SupportsMetadata(Protocol):
    def set_metadata(self, config: DictConfig, model_name: str) -> None: ...


@runtime_checkable
class SupportsVideoLogging(Protocol):
    def log_video(
        self, key: str, videos: list[Any], step: int | None = None, **kwargs: Any
    ) -> None: ...
