from torch import nn

from icenet_mp.models.common import Mask, RestrictRange
from icenet_mp.types import DataSpace, RangeRestriction, TensorNCHW, TensorNTCHW


class BaseDecoder(nn.Module):
    """Decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)
    """

    def __init__(
        self,
        *,
        data_space_in: DataSpace,
        data_space_out: DataSpace,
        mask_dir: str | None = None,
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

        # Load the requested mask (ACTIVE/LAND/NONE)
        self.mask = Mask(
            mask_type=mask_type,
            output_shape=self.data_space_out.shape,
            mask_dir=mask_dir,
        )

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
        return self.mask(self.restrict(x))

    def rollout(self, x: TensorNTCHW, persistence: TensorNTCHW | None) -> TensorNTCHW:
        """Decode latent space into output space across multiple timesteps.

        The default implementation simply calls `self.forward` on each time slice
        simultaneously by reshaping the input to combine the batch and time dimensions,
        before reshaping back. The shared last-gating steps are applied
        in rollout via finalise(), so concrete decoders only implement forward().

        Note that this (rollout method) also increases the effective batch size for any batch
        normalisation layers in the encoder.

        Args:
            x: TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total, latent_height, latent_width)
            persistence: TensorNCHW with (batch_size, 1, output_channels, output_height, output_width) to add to every forecast step

        Returns:
            TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)

        """
        # Although this should only be called with n_forecast_steps timeslices, we can
        # make the decoder more generic by simply reading the number of timeslices from
        # the input.
        batch_size, n_timeslices = x.shape[0], x.shape[1]
        # Pass the latents through the decoder to return to output space
        output_nchw = self(x.reshape(-1, *self.data_space_in.chw))
        if persistence is not None:
            # Expand persistence to match output shape and add to every forecast step
            persistence = persistence.expand(-1, n_timeslices, -1, -1, -1)
            persistence_nchw = persistence.reshape(-1, *self.data_space_out.chw)
            output_nchw += persistence_nchw
        # Add persistence to the output and finalise (bound and mask) the result
        output = self.finalise(output_nchw)
        # Reshape back to [batch, n_timeslices, C_out, H_out, W_out]
        return output.reshape(batch_size, n_timeslices, *self.data_space_out.chw)
