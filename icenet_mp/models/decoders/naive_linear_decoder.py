from typing import Any

from torch import nn
from torch.nn.functional import sigmoid

from icenet_mp.models.common import ResizingInterpolation
from icenet_mp.types import TensorNCHW

from .base_decoder import BaseDecoder


class NaiveLinearDecoder(BaseDecoder):
    """Naive, linear decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)
    """

    def __init__(self, *, bounded: bool = False, **kwargs: Any) -> None:
        """Initialise a NaiveLinearDecoder."""
        super().__init__(**kwargs)

        # specify whether the output is bounded between 0 and 1
        self.bounded = bounded

        # List of layers
        layers: list[nn.Module] = []

        # Convolve to the desired number of output channels
        layers.append(
            nn.Conv2d(self.data_space_in.channels, self.data_space_out.channels, 1)
        )

        # Resize to the desired output shape
        if self.data_space_in.shape != self.data_space_out.shape:
            layers.append(ResizingInterpolation(self.data_space_out.shape))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space with a linear transform.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        if self.bounded:
            return sigmoid(self.model(x))
        return self.model(x)
