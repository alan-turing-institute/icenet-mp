from enum import StrEnum


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


class SkipConnectionType(StrEnum):
    """Enum for decoder skip connection types."""

    ADDITIVE = "additive"
    CONVOLUTIONAL = "convolutional"
    GATED = "gated"
    NONE = "none"
