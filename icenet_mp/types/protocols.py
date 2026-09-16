from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SupportsImageLogging(Protocol):
    def log_image(
        self, key: str, images: list[Any], step: int | None = None, **kwargs: Any
    ) -> None: ...


@runtime_checkable
class SupportsVideoLogging(Protocol):
    def log_video(
        self, key: str, videos: list[Any], step: int | None = None, **kwargs: Any
    ) -> None: ...
