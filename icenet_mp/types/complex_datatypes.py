from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Self, cast

from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .typedefs import DiffMode, DiffStrategy, TensorNTCHW


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
class ModelStepOutput(Mapping[str, Tensor]):
    """Output of a model step: prediction, target, and loss."""

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
        msg = f"Key {key} not found in ModelStepOutput"
        raise KeyError(msg)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys of ModelStepOutput."""
        yield "prediction"
        yield "target"
        yield "loss"

    def __len__(self) -> int:
        """Return ModelStepOutput length."""
        return 3

    def copy(self) -> dict[str, Tensor]:
        """Return a plain dict copy, required by Lightning's manual optimization loop."""
        return dict(self)


@dataclass
class PlotSpec:
    """Configure how sea-ice plots are rendered.

    Attributes:
        variable: Variable name shown in plots / used for routing.
        title_groundtruth: Title above the ground-truth panel.
        title_prediction: Title above the prediction panel.
        title_difference: Title above the difference panel.
        n_contour_levels: Number of contour levels per panel.
        colourmap: colourmap used for GT/prediction panels.
        dpi: Dots per inch for figure rendering (default 300).
        include_difference: Whether to draw a difference panel.
        diff_mode: Difference definition (e.g. "signed", "absolute", "smape").
        diff_strategy: Strategy for animations (precompute, two-pass, per-frame).
        selected_timestep: Slice index when a single timestep is needed.
        vmin: Lower bound for GT/prediction colour scale (None = infer).
        vmax: Upper bound for GT/prediction colour scale (None = infer).
        colourbar_location: "vertical" or "horizontal".
        colourbar_strategy: "shared" or "separate" colourbars.
        outside_warn: Threshold for “values outside display range” warnings.
        severe_outside: Severe threshold for clipping warnings.
        include_shared_range_mismatch_check: If True, add magnitude mismatch nudges.

    """

    variable: str = "sea_ice_concentration"
    title_groundtruth: str = "Ground Truth"
    title_prediction: str = "Prediction"
    title_difference: str = "Difference"

    n_contour_levels: int = 51
    colourmap: str = "viridis"
    dpi: int = 300

    # Difference pane
    include_difference: bool = True
    diff_mode: DiffMode = "signed"
    diff_strategy: DiffStrategy = "precompute"
    selected_timestep: int = 0

    # Colourscale ranges: defaults to [0,1]
    vmin: float | None = 0.0
    vmax: float | None = 1.0

    # Colourbar layout
    colourbar_location: Literal["vertical", "horizontal"] = "horizontal"
    colourbar_strategy: Literal["shared", "separate"] = "shared"

    # Range Check/warnings in badge
    outside_warn: float = 0.05
    severe_outside: float = 0.20
    include_shared_range_mismatch_check: bool = True

    # Optional metadata for titling
    # hemisphere: "north" | "south" when known (used in titles)
    hemisphere: Literal["north", "south"] | None = None
    # metadata_subtitle: free-form text (e.g., "epochs=50; train=2010-2018")
    metadata_subtitle: str | None = None

    # Footer control
    include_footer_metadata: bool = True

    # Video settings
    video_fps: int = 2
    video_format: Literal["mp4", "gif"] = "mp4"

    # Per-variable styles
    per_variable_styles: dict[str, dict[str, str | float | bool]] = field(
        default_factory=lambda: {
            # Sea ice concentration
            "sic-icenet:ice_conc": {"cmap": "Blues_r"},
            "sic-ssmis:ice_conc": {"cmap": "Blues_r"},
        }
    )

    def __add__(
        self, other: "PlotSpec | DictConfig | dict[str, Any] | None"
    ) -> "PlotSpec":
        """Combine two PlotSpec instances or a PlotSpec with a dictionary."""
        if other is None:
            return self
        if isinstance(other, PlotSpec):
            dict_other = asdict(other)
        elif isinstance(other, DictConfig):
            dict_other = cast("dict[str, Any]", OmegaConf.to_container(other))
        else:
            dict_other = dict(other)
        return PlotSpec(**(asdict(self) | dict_other))
