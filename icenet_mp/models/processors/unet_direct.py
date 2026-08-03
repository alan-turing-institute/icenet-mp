"""One-shot (non-autoregressive) UNet processor.

Description:
    Predicts every forecast step in a single forward pass, instead of the default
    BaseProcessor.rollout behaviour of calling forward once per forecast step and
    feeding each prediction back in as history. Same rationale as VitDirectProcessor:
    removes the n_forecast_steps-times cost of running the network repeatedly per
    training step, at the price of no longer conditioning each forecast day on the
    model's own previously predicted days (only on the original history window).
"""

from typing import Any

from torch import nn

from icenet_mp.types import ProcessorOutput, TensorNCHW, TensorNTCHW

from .unet import UNetProcessor


class UNetDirectProcessor(UNetProcessor):
    """UNet processor that predicts all forecast steps in one pass."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialise a UNetDirectProcessor.

        Args:
            kwargs: Arguments to UNetProcessor (kernel_size, norm_type,
                start_out_channels, plus BaseProcessor's arguments).

        """
        super().__init__(**kwargs)

        # Replace the final layer built by UNetProcessor.__init__: project to every
        # forecast step at once (out_channels * n_forecast_steps) instead of a single
        # next timestep. Input channel count is unchanged, so read it back from the
        # layer UNetProcessor already built rather than re-deriving start_out_channels.
        in_channels = self.final_layer.in_channels
        self.final_layer = nn.Conv2d(
            in_channels,
            self.data_space.channels * self.n_forecast_steps,
            kernel_size=1,
            padding="same",
        )

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:  # noqa: ARG002
        """Predict all forecast steps in a single forward pass (no autoregression).

        Args:
            x: Encoded input TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
            y: Unused; accepted for interface compatibility with BaseProcessor.rollout.

        Returns:
            ProcessorOutput with prediction TensorNTCHW (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width).

        """
        batch, n_history, channels, height, width = x.shape
        # Equivalent to BaseProcessor.rollout's `cat(list(window), dim=1)`: history
        # steps ordered oldest to newest, each contributing `channels` channels.
        window_cat = x.reshape(batch, n_history * channels, height, width)
        output: TensorNCHW = self(
            window_cat
        )  # (batch, n_forecast_steps * channels, height, width)
        output = output.reshape(batch, self.n_forecast_steps, channels, height, width)
        return ProcessorOutput(prediction=output)
