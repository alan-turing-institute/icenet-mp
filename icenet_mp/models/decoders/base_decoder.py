from pathlib import Path

import numpy as np
from torch import Tensor, from_numpy, nn
from torch.nn.functional import sigmoid

from icenet_mp.types import DataSpace, TensorNCHW, TensorNTCHW


class BaseDecoder(nn.Module):
    """Decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
    """

    # buffer in __init__, annotated here to make the type explicitly
    mask: Tensor

    def __init__(
        self,
        *,
        data_space_in: DataSpace,
        data_space_out: DataSpace,
        n_forecast_steps: int,
        active_mask_path: str | None = None,
        use_mask: bool = False,
        bounded: bool = False,
    ) -> None:
        """Initialise a BaseDecoder."""
        super().__init__()
        self.data_space_in = data_space_in
        self.data_space_out = data_space_out
        self.n_forecast_steps = n_forecast_steps
        self.name = data_space_out.name

        # Whether to bound the output between 0 and 1
        self.bounded = bounded

        # Load the active mask only when requested. When off, finalise() skips
        # the multiply entirely. Path is derived from the dataset (ref CommonDataModule.active_mask_path)
        # Only require the file to exist when use_mask is True, fail loudly if not.
        self.use_mask = use_mask
        if use_mask:
            if active_mask_path is None or not Path(active_mask_path).exists():
                msg = (
                    f"use_mask is enabled but no active mask was found at "
                    f"{active_mask_path}. Masks are generated per dataset during "
                    f"`datasets create` (currently for SSMIS datasets)."
                )
                raise FileNotFoundError(msg)
            mask = from_numpy(np.load(Path(active_mask_path))).float()
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

        Masking is applied AFTER bounding so masked cells are exactly 0 regardless of
        `bounded` (applying it before would leak sigmoid(0)=0.5 into masked cells).
        Every concrete decoder should call this at the end of its `forward`.
        """
        if self.bounded:
            x = sigmoid(x)
        if self.use_mask:
            x = x * self.mask.to(dtype=x.dtype)
        return x

    def rollout(self, x: TensorNTCHW) -> TensorNTCHW:
        """Decode latent space into output space across multiple timesteps.

        The default implementation simply calls `self.forward` on each time slice
        simultaneously by reshaping the input to combine the batch and time dimensions,
        before reshaping back.

        Note that this also increases the effective batch size for any batch
        normalisation layers in the encoder.

        Args:
            x: TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)

        """
        return self(x.reshape(-1, *self.data_space_in.chw)).reshape(
            -1, self.n_forecast_steps, *self.data_space_out.chw
        )
