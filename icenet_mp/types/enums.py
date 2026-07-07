from enum import IntEnum, StrEnum
from typing import Any


class BetaSchedule(StrEnum):
    """Enum for diffusion beta schedule types."""

    LINEAR = "linear"
    COSINE = "cosine"


class MaskType(StrEnum):
    """Enum for types of masking."""

    ACTIVE = "active"
    LAND = "land"
    NONE = "none"


class RangeRestriction(StrEnum):
    """Enum for bounded output types."""

    CLAMP = "clamp"
    NONE = "none"
    SIGMOID = "sigmoid"
    TANH = "tanh"

    @classmethod
    def _missing_(cls, value: Any) -> "RangeRestriction":  # noqa: ARG003,ANN401
        """Handle missing values by returning NONE."""
        return cls.NONE


class TensorDimensions(IntEnum):
    """Enum for tensor dimensions."""

    THW = 3  # Time, Height, Width
    BTCHW = 5  # Batch, Time, Channels, Height, Width
