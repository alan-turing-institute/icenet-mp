from pathlib import Path

import numpy as np
from torch import Tensor, from_numpy, nn

from icenet_mp.models.common import RestrictRange
from icenet_mp.types import DataSpace, RangeRestriction, TensorNCHW, TensorNTCHW


class BaseDecoder(nn.Module):
    """Decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)
    """

    # buffer in __init__, annotated here to make the type explicitly
    mask: Tensor

    def __init__(
        self,
        *,
        active_mask_path: str | None = None,
        data_space_in: DataSpace,
        data_space_out: DataSpace,
        land_mask_path: str | None = None,
        mask_type: str | None = None,
        restrict_range: str = "none",
    ) -> None:
        """Initialise a BaseDecoder."""
        super().__init__()
        self.data_space_in = data_space_in
        self.data_space_out = data_space_out
        self.name = data_space_out.name

        # Bound (or not) the output into [0, 1], select: none/sigmoid/clamp/tanh.
        self.restrict = RestrictRange(
            RangeRestriction(restrict_range), min_val=0, max_val=1
        )

        # Load the active mask only when requested. When off, finalise() skips
        # the multiply entirely. Path is derived from the dataset (ref CommonDataModule.active_mask_path)
        # Require the file to exist when mask_type is defined, fail loudly if not.
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
            if tuple(mask.shape) != self.data_space_out.shape:
                msg = (
                    f"{mask_type} mask shape {tuple(mask.shape)} does not match "
                    f"decoder output shape {self.data_space_out.shape}."
                )
                raise ValueError(msg)
            self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space for a single timestep.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        msg = "If you are using the default rollout method, you must implement forward."
        raise NotImplementedError(msg)

    def finalise(self, x: TensorNCHW) -> TensorNCHW:
        """Apply shared output steps: bound if requested, then zero masked cells.

        Masking is applied AFTER the range restriction so masked cells are exactly 0
        regardless of bounding (applying it before would leak e.g. sigmoid(0)=0.5 into
        masked cells). Called once by `rollout` after the per-frame `forward`, so every
        decoder gets it automatically without having to call it themselves.

        RangeRestriction choices are: none/sigmoid/tanh/clamp
        """
        x = self.restrict(x)
        if self.use_mask:
            x = x * self.mask.to(dtype=x.dtype)
        return x

    def rollout(self, x: TensorNTCHW) -> TensorNTCHW:
        """Decode latent space into output space across multiple timesteps.

        The default implementation simply calls `self.forward` on each time slice
        simultaneously by reshaping the input to combine the batch and time dimensions,
        before reshaping back. The shared last-gating steps are applied
        in rollout via finalise(), so concrete decoders only implement forward().

        Note that this (rollout method) also increases the effective batch size for any batch
        normalisation layers in the encoder.

        Args:
            x: TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)

        """
        # Although this should only be called with n_forecast_steps timeslices, we can
        # make the decoder more generic by simply reading the number of timeslices from
        # the input.
        batch_size, n_timeslices = x.shape[0], x.shape[1]
        output = self.finalise(self(x.reshape(-1, *self.data_space_in.chw)))
        return output.reshape(batch_size, n_timeslices, *self.data_space_out.chw)
