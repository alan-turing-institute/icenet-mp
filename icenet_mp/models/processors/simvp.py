from typing import Any

from torch import nn

from icenet_mp.types import TensorNCHW

from .base_processor import BaseProcessor


class TemporalMixerBlock(nn.Module):
    """Residual spatial-temporal mixing block for flattened history features."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        """Initialise a depthwise spatial mixer followed by channel mixing."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding="same",
                groups=channels,
            ),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Mix features while preserving shape through a residual path."""
        return x + self.block(x)


class SimVPProcessor(BaseProcessor):
    """SimVP-style processor for convolutional spatio-temporal forecasting.

    Each history frame is first embedded independently in space. The encoded history
    is then flattened along the channel axis and mixed by residual convolutional blocks
    before being decoded to the next latent frame. The standard ``BaseProcessor``
    rollout applies this one-step model autoregressively for multiple forecast steps.
    """

    def __init__(
        self,
        *,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        n_mixer_blocks: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialise the SimVP-style processor."""
        super().__init__(**kwargs)
        if hidden_channels <= 0:
            msg = "hidden_channels must be greater than 0."
            raise ValueError(msg)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            msg = "kernel_size must be a positive odd integer."
            raise ValueError(msg)
        if n_mixer_blocks <= 0:
            msg = "n_mixer_blocks must be greater than 0."
            raise ValueError(msg)

        self.hidden_channels = hidden_channels
        history_channels = hidden_channels * self.n_history_steps

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(
                self.data_space.channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.SiLU(),
        )
        self.temporal_mixer = nn.Sequential(
            *(
                TemporalMixerBlock(history_channels, kernel_size)
                for _ in range(n_mixer_blocks)
            )
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(history_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, self.data_space.channels, kernel_size=1),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Predict one latent frame from a concatenated history window."""
        batch_size, channels, height, width = x.shape
        expected_channels = self.data_space.channels * self.n_history_steps
        if channels != expected_channels:
            msg = (
                f"Expected {expected_channels} concatenated history channels, "
                f"got {channels}."
            )
            raise ValueError(msg)

        history = x.reshape(
            batch_size,
            self.n_history_steps,
            self.data_space.channels,
            height,
            width,
        )
        encoded = self.spatial_encoder(
            history.reshape(
                batch_size * self.n_history_steps,
                self.data_space.channels,
                height,
                width,
            )
        )
        encoded = encoded.reshape(
            batch_size,
            self.n_history_steps * self.hidden_channels,
            height,
            width,
        )
        return self.decoder(self.temporal_mixer(encoded))
