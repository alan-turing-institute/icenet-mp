"""CNN encoder support for conditioning data with missing values."""

from typing import Any

import torch
from torch import nn

from icenet_mp.types import TensorNCHW

from .cnn_encoder import CNNEncoder


class MissingDataCNNEncoder(CNNEncoder):
    """CNN encoder that preserves an explicit signal for missing observations.

    Non-finite values are replaced with zero before convolution and a per-pixel
    availability channel is appended. A trainable 1x1 adapter maps the augmented
    input back to the original channel count so the existing CNN architecture and
    latent-space contract remain unchanged.

    During training, complete time slices can also be dropped at random. Since
    ``BaseEncoder.rollout`` folds batch and time together before calling ``forward``,
    this conditioning dropout is sampled independently for each history time slice.
    """

    def __init__(
        self,
        *,
        conditioning_dropout_probability: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialise the missing-data-aware CNN encoder."""
        if not 0.0 <= conditioning_dropout_probability <= 1.0:
            msg = "conditioning_dropout_probability must be between 0 and 1."
            raise ValueError(msg)

        super().__init__(**kwargs)
        self.conditioning_dropout_probability = conditioning_dropout_probability

        channels = self.data_space_in.channels
        self.input_adapter = nn.Conv2d(channels + 1, channels, kernel_size=1)

        # Start as an identity mapping over the observed data channels. The
        # availability channel begins with zero weight and is learned only if useful.
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            for channel in range(channels):
                self.input_adapter.weight[channel, channel, 0, 0] = 1.0
            if self.input_adapter.bias is not None:
                self.input_adapter.bias.zero_()

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Encode one time slice while retaining an observation-availability signal."""
        finite = torch.isfinite(x)
        availability = finite.to(dtype=x.dtype).mean(dim=1, keepdim=True)
        clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.training and self.conditioning_dropout_probability > 0.0:
            drop_mask = (
                torch.rand(
                    (x.shape[0], 1, 1, 1),
                    device=x.device,
                )
                < self.conditioning_dropout_probability
            )
            clean = clean.masked_fill(drop_mask, 0.0)
            availability = availability.masked_fill(drop_mask, 0.0)

        augmented = torch.cat((clean, availability), dim=1)
        return super().forward(self.input_adapter(augmented))
