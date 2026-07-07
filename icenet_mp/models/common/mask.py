from pathlib import Path

import numpy as np
from torch import Tensor, from_numpy, nn


class Mask(nn.Module):
    """Mask out inactive/land cells in model output.

    Loads a mask from disk and applies it multiplicatively. If no mask is requested then
    the input is returned unchanged.
    """

    def __init__(
        self,
        *,
        mask_type: str | None,
        output_shape: tuple[int, ...],
        mask_dir: str | Path | None = None,
    ) -> None:
        """Initialise a Mask.

        Args:
            mask_type: "active" (active+land), "land" (land only), or None to disable.
            output_shape: Expected (channels, height, width) shape of the tensor this
                mask will be applied to, used to validate the mask loaded from disk.
            mask_dir: Directory holding `active_mask.npy`/`land_mask.npy`, generated for
                SSMIS datasets by `datasets create`. Required when `mask_type` is
                "active" or "land".

        """
        super().__init__()
        if mask_type is None:
            return
        if mask_type not in ("active", "land"):
            msg = f"Unknown mask_type {mask_type!r}; expected one of active/land/None."
            raise ValueError(msg)

        mask_path = Path(mask_dir) / f"{mask_type}_mask.npy" if mask_dir else None
        if mask_path is None or not mask_path.exists():
            msg = (
                f"{mask_type} mask is requested but no mask was found at {mask_path}. "
                "Masks are generated for SSMIS datasets during `datasets create`."
            )
            raise FileNotFoundError(msg)
        mask = from_numpy(np.load(mask_path)).float()
        if tuple(mask.shape) != tuple(output_shape):
            msg = (
                f"{mask_type} mask shape {tuple(mask.shape)} does not match expected "
                f"output shape {tuple(output_shape)}."
            )
            raise ValueError(msg)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Multiply by the mask if loaded, otherwise return x unchanged."""
        mask = getattr(self, "mask", None)
        if mask is None:
            return x
        return x * mask.to(dtype=x.dtype)
