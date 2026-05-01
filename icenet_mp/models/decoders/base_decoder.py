from torch import nn

from icenet_mp.types import DataSpace, TensorNCHW, TensorNTCHW


class BaseDecoder(nn.Module):
    """Decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
    """

    def __init__(
        self,
        *,
        data_space_in: DataSpace,
        data_space_out: DataSpace,
        n_forecast_steps: int,
    ) -> None:
        """Initialise a BaseDecoder."""
        super().__init__()
        self.data_space_in = data_space_in
        self.data_space_out = data_space_out
        self.n_forecast_steps = n_forecast_steps
        self.name = data_space_out.name

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space for a single timestep.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        msg = "If you are using the default rollout method, you must implement forward."
        raise NotImplementedError(msg)

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
