from collections.abc import Callable
from functools import cached_property

from torch import nn

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
        latitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        longitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
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
        self.latitudes_fn = latitudes_fn
        self.longitudes_fn = longitudes_fn
        self.name = data_space_in.name
        self.n_history_steps = n_history_steps

    @cached_property
    def latitudes(self) -> dict[str, list[float]]:
        return {} if not self.latitudes_fn else self.latitudes_fn()

    @cached_property
    def longitudes(self) -> dict[str, list[float]]:
        return {} if not self.longitudes_fn else self.longitudes_fn()

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

        The default implementation simply calls `self.forward` on each time slice
        simultaneously by reshaping the input to combine the batch and time dimensions,
        before reshaping back.

        Note that this also increases the effective batch size for any batch
        normalisation layers in the encoder.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

        Returns:
            TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)

        """
        return self(x.reshape(-1, *self.data_space_in.chw)).reshape(
            -1, self.n_history_steps, *self.data_space_out.chw
        )
