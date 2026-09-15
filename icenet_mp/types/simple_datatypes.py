from dataclasses import dataclass
from typing import NamedTuple, TypedDict

from anemoi.datasets.create.recipe import Recipe
from matplotlib.colors import Normalize
from torch import Tensor

from .typedefs import TensorNTCHW


@dataclass
class AnemoiCleanupArgs:
    """Arguments for anemoi cleanup."""

    path: str
    command: str = "unused"


class AnemoiDatasetStatus(NamedTuple):
    """Status of an Anemoi dataset."""

    copy_in_progress: bool
    download_complete: bool
    is_finalised: bool


@dataclass
class AnemoiFinaliseArgs:
    """Arguments for anemoi finalise."""

    path: str
    recipe: Recipe
    command: str = "unused"


@dataclass
class AnemoiInitArgs:
    """Arguments for anemoi init."""

    path: str
    recipe: Recipe
    command: str = "unused"
    overwrite: bool = False


@dataclass
class AnemoiInspectArgs:
    """Arguments for anemoi inspect."""

    detailed: bool
    path: str
    progress: bool
    size: bool
    statistics: bool


@dataclass
class AnemoiLoadArgs:
    """Arguments for anemoi load."""

    path: str
    recipe: Recipe
    command: str = "unused"


class DataloaderArgs(TypedDict):
    """Arguments for the data loader."""

    batch_sampler: None
    batch_size: int
    drop_last: bool
    num_workers: int
    prefetch_factor: int | None
    persistent_workers: bool
    sampler: None
    worker_init_fn: None


class DiffColourmap(NamedTuple):
    """Specify the colour scale used for a difference panel.

    Attributes:
        norm: Normalisation for mapping values to colours (e.g. TwoSlopeNorm for signed diffs).
        vmin: Lower bound if no norm is provided.
        vmax: Upper bound if no norm is provided.
        cmap: Matplotlib colourmap name.

    """

    norm: Normalize | None
    vmin: float | None
    vmax: float | None
    cmap: str


@dataclass(frozen=True)
class Metadata:
    """Structured metadata extracted from training configuration.

    Attributes:
        model: Model name (if available).
        current_epoch: Current training epoch (if available).
        start: Training start date string (if available).
        end: Training end date string (if available).
        cadence: Training data cadence string (if available).
        n_points: Number of training points calculated from date range and cadence.
        vars_by_source: Dictionary mapping dataset source names to lists of variable names.
        n_history_steps: Number of history steps used as model input window (days).

    """

    model: str | None = None
    current_epoch: int | None = None
    start: str | None = None
    end: str | None = None
    cadence: str | None = None
    n_points: int | None = None
    n_history_steps: int | None = None
    vars_by_source: dict[str, list[str]] | None = None


@dataclass(frozen=True)
class ProcessorOutput:
    """Output of a processor rollout step."""

    prediction: TensorNTCHW
    loss: Tensor | None = None


@dataclass(frozen=True)
class VariableStyle:
    """Styling configuration for individual variables.

    Attributes:
        cmap: Matplotlib colourmap name (e.g., "viridis", "RdBu_r").
        vmin: Minimum value for colour scale.
        vmax: Maximum value for colour scale.
        units: Display units for the variable (e.g., "K", "m/s").

    """

    cmap: str
    vmin: float | None = None
    vmax: float | None = None
    units: str | None = None
