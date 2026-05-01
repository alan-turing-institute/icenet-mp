from typing import Literal

from jaxtyping import Float, Int
from numpy import float32, int64
from numpy.typing import NDArray
from torch import Tensor

# Numpy arrays
ArrayHW = Float[NDArray[float32], "height width"]
ArrayHWV = Float[NDArray[float32], "height width variable"]
ArrayCHW = Float[NDArray[float32], "channels height width"]
ArrayTHW = Float[NDArray[float32], "time height width"]
ArrayTCHW = Float[NDArray[float32], "time channels height width"]
ArrayIndices2D = Int[NDArray[int64], "height width"]

# PyTorch tensors
TensorNCHW = Float[Tensor, "batch channels height width"]
TensorNTCHW = Float[Tensor, "batch time channels height width"]

# DiffMode: what you compute
# - "signed": target - prediction (can be ±, so symmetric colour scale around 0)
# - "absolute": |target - prediction| (≥ 0, sequential scale)
# - "smape": |pred - target| / ((|pred|+|target|)/2) (≥ 0, sequential scale)
DiffMode = Literal["signed", "absolute", "smape"]

# DiffStrategy: when you compute (for animations)
# - "precompute": compute full diff stream once (fast playback, more RAM)
# - "two-pass": scan once to figure the scale, compute per-frame (balanced)
# - "per-frame": compute per-frame (low RAM, more CPU)
DiffStrategy = Literal["precompute", "two-pass", "per-frame"]

Hemisphere = Literal["north", "south"]
