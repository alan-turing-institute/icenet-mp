from pathlib import Path

import numpy as np
from torch import Tensor, from_numpy, nn


class Mask(nn.Module):
    """Mask out inactive/land cells in model output.

    Loads a mask from disk and applies it as a multiplicatively.
    """

    mask: Tensor

    def __init__(
        self,
        *,
        mask_type: str | None,
        output_shape: tuple[int, ...],
        active_mask_path: str | None = None,
        land_mask_path: str | None = None,
    ) -> None:
        """Initialise a Mask.

        Args:
            mask_type: "active" (active+land), "land" (land only), or
                "none"/None to disable masking.
            output_shape: Expected (channels, height, width) shape of the
                tensor this mask will be applied to, used to validate the
                mask loaded from disk.
            active_mask_path: Path to the active mask. Used when `mask_type`
                is "active".
            land_mask_path: Path to the land mask. Used when `mask_type` is
                "land".

        """
        super().__init__()
        if mask_type not in (None, "none", "active", "land"):
            msg = f"Unknown mask_type {mask_type!r}; expected one of none/active/land."
            raise ValueError(msg)
        self.mask_type = mask_type
        self.use_mask = mask_type in ("active", "land")

        if self.use_mask:
            mask_path = active_mask_path if mask_type == "active" else land_mask_path
            if mask_path is None or not Path(mask_path).exists():
                msg = (
                    f"{mask_type} mask is requested but no mask was found at "
                    f"{mask_path}. Masks are generated per dataset during "
                    f"`datasets create` (currently for SSMIS datasets)."
                )
                raise FileNotFoundError(msg)
            mask = from_numpy(np.load(Path(mask_path))).float()
            if tuple(mask.shape) != tuple(output_shape):
                msg = (
                    f"{mask_type} mask shape {tuple(mask.shape)} does not match "
                    f"expected output shape {tuple(output_shape)}."
                )
                raise ValueError(msg)
            self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Multiply by the mask if enabled, otherwise return x unchanged."""
        if self.use_mask:
            return x * self.mask.to(dtype=x.dtype)
        return x
