from torch import nn, stack

from icenet_mp.types import DataSpace, TensorNCHW, TensorNTCHW


class BaseEncoder(nn.Module):
    """Encoder that takes data in an input space and translates it to a smaller latent space.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        data_space_in: DataSpace,
        latent_space: tuple[int, int],
        latitudes: dict[str, list[float]],
        longitudes: dict[str, list[float]],
        n_history_steps: int,
    ) -> None:
        """Initialise a BaseEncoder."""
        super().__init__()
        self.data_space_in = data_space_in
        self.data_space_out = DataSpace(
            name=f"latent_space_{data_space_in.name}",
            channels=self.data_space_in.channels,
            shape=latent_space,
        )
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.name = data_space_in.name
        self.n_history_steps = n_history_steps

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space for a single timestep.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        msg = "If you are using the default rollout method, you must implement forward."
        raise NotImplementedError(msg)

    def rollout(self, x: TensorNTCHW) -> TensorNTCHW:
        """Encode input space into latent space across multiple timesteps.

        The default implementation simply calls `self.forward` independently on each
        time slice. These are then stacked together to produce the final output.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

        Returns:
            TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)

        """
        return stack(
            [
                # Rollout the model over the input slices, producing an output for each one.
                self(x[:, idx_t, :, :, :])
                for idx_t in range(self.n_history_steps)
            ],
            dim=1,
        )
