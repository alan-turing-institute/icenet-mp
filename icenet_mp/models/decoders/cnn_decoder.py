import logging
from typing import Any

from torch import nn
from torch.nn.functional import sigmoid

from icenet_mp.models.common import ConvBlockUpsample, ResizingInterpolation
from icenet_mp.types import TensorNCHW

from .base_decoder import BaseDecoder

logger = logging.getLogger(__name__)


class CNNDecoder(BaseDecoder):
    """Decoder that uses a convolutional neural net (CNN) to translate latent space back to data space.

    - Increase size with interpolation (if needed)
    - n_layers of size-increasing convolutional blocks
    - Decrease size with interpolation (if needed)
    - Convolve to number of output channels (if needed)

    Input space:
        TensorNTCHW with (batch_size, n_timeslices, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        activation: str = "ReLU",
        bounded: bool = False,
        kernel_size: int = 3,
        n_layers: int = 3,
        n_subblocks: int = 2,
        scale_factor: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialise a CNNDecoder."""
        super().__init__(**kwargs)

        # specify whether the output is bounded between 0 and 1
        self.bounded = bounded

        # Calculate the factor by which the scale changes after n_layers
        layer_factor = scale_factor**n_layers

        # Ensure number of channels is divisible by the power of two implied by n_layers
        if self.data_space_in.channels % layer_factor:
            msg = (
                f"The number of input channels {self.data_space_in.channels} must be divisible by {layer_factor}. "
                f"Without this, it is not possible to apply {n_layers} convolutions."
            )
            raise ValueError(msg)

        # Calculate the smallest input shape that would produce an output at least as
        # large as the desired output shape. Note that this may not be exact, since we
        # double the size at each layer.
        minimal_input_shape = (
            -(self.data_space_out.shape[0] // -layer_factor),
            -(self.data_space_out.shape[1] // -layer_factor),
        )

        # Construct list of layers
        layers: list[nn.Module] = []
        logger.debug("CNNDecoder (%s) with %d layers", self.name, n_layers)

        # If necessary, resize until we reach the minimal input shape. This ensures that
        # the post-convolution shape will be at least as large as the desired output
        # shape so any further resizing will be a size decrease.
        shape = (
            max(minimal_input_shape[0], self.data_space_in.shape[0]),
            max(minimal_input_shape[1], self.data_space_in.shape[1]),
        )
        if shape != self.data_space_in.shape:
            layers.append(ResizingInterpolation(shape))
            logger.debug(
                "- ResizingInterpolation from %s to %s",
                self.data_space_in.shape,
                shape,
            )

        # Add n_layers size-increasing convolutional blocks
        n_channels = self.data_space_in.channels
        for _ in range(n_layers):
            layers.append(
                ConvBlockUpsample(
                    n_channels,
                    activation=activation,
                    kernel_size=kernel_size,
                    n_subblocks=n_subblocks,
                    scale_factor=scale_factor,
                )
            )
            logger.debug(
                "- ConvBlockUpsample (%s, %s) with %d channels",
                activation,
                kernel_size,
                n_channels,
            )
            n_channels //= scale_factor
            shape = (shape[0] * scale_factor, shape[1] * scale_factor)

        # If necessary, resize downwards to match the output shape
        if shape != self.data_space_out.shape:
            layers.append(ResizingInterpolation(self.data_space_out.shape))
            logger.debug(
                "- ResizingInterpolation from %s to %s",
                shape,
                self.data_space_out.shape,
            )

        # Run a final convolution that will both ensure that we have the right number of
        # output channels and also the the final operation is a layer that can produce
        # values across the full output range (not just the range of the activation
        # function in the last ConvBlockUpsample).
        layers.append(nn.Conv2d(n_channels, self.data_space_out.channels, 1))
        logger.debug(
            "- Channel convolution from %d to %d",
            n_channels,
            self.data_space_out.channels,
        )

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space with a CNN.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        if self.bounded:
            return sigmoid(self.model(x))
        return self.model(x)
