"""Processor based on the translator from SimVPv2.

Reference:
    SimVP: Simpler yet Better Video Prediction
    (Gao et al., 2022) [https://arxiv.org/abs/2206.05099]
    SimVPv2: Towards Simple yet Powerful Spatiotemporal Predictive Learning
    (Tan et al., 2022) [https://arxiv.org/abs/2211.12509]

We take the temporal translator from SimVPv2: a flat stack of "gSTA" (Gated
Spatiotemporal Attention) blocks replacing the original SimVP Inception-based U-Net
structure.

Our implementation of GatedAttentionBlock is slightly modified to allow different
numbers of input and output channels.
"""

from typing import Any

from torch import nn

from icenet_mp.models.common import GatedAttentionBlock
from icenet_mp.types import ProcessorOutput, TensorNCHW, TensorNTCHW

from .base_processor import BaseProcessor

class SimVPProcessor(BaseProcessor):
    """Processor based on the translator from SimVPv2.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dilation: int = 3,
        drop_path_prob: float = 0.1,
        hid_channels_spatial: int = 64,
        hid_channels_temporal: int = 256,
        kernel_size: int = 21,
        mlp_drop_prob: float = 0.0,
        mlp_ratio: float = 4.0,
        temporal_depth: int = 8,
        **kwargs: Any,
    ) -> None:
        """Initialise a SimVPProcessor.

        Args:
            dilation: Dilation used by gated attention blocks in the translator.
            drop_path_prob: Probability with which to drop the residual branch of each
                gated attention block in the translator during training.
            hid_channels_spatial: Hidden channel count for the spatial encoder/decoder.
            hid_channels_temporal: Hidden channel count for the temporal translator.
            kernel_size: Kernel size for gated attention blocks in the translator.
            mlp_drop_prob: Element-wise dropout probability for the translator conv-MLP blocks.
            mlp_ratio: Hidden-channel expansion ratio for the translator conv-MLP blocks.
            temporal_depth: Number of gated attention blocks in the temporal translator.
            kwargs: Arguments to BaseProcessor.

        """
        super().__init__(**kwargs)
        if temporal_depth < 2:  # noqa: PLR2004
            msg = f"temporal_depth must be at least 2 to project in_channels -> hid_channels -> out_channels, got {temporal_depth}"
            raise ValueError(msg)

        # Translator: a flat stack of gSTA blocks, with time folded into the channels
        self.translator = nn.Sequential(
            GatedAttentionBlock(
                self.n_history_steps * hid_channels_spatial,
                hid_channels_temporal,
                dilation=dilation,
                drop_path_prob=drop_path_prob,
                kernel_size=kernel_size,
                mlp_drop_prob=mlp_drop_prob,
                mlp_ratio=mlp_ratio,
            ),
            *(
                GatedAttentionBlock(
                    hid_channels_temporal,
                    hid_channels_temporal,
                    dilation=dilation,
                    drop_path_prob=drop_path_prob,
                    kernel_size=kernel_size,
                    mlp_drop_prob=mlp_drop_prob,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(temporal_depth - 2)
            ),
            GatedAttentionBlock(
                hid_channels_temporal,
                self.n_forecast_steps * hid_channels_spatial,
                dilation=dilation,
                drop_path_prob=drop_path_prob,
                kernel_size=kernel_size,
                mlp_drop_prob=mlp_drop_prob,
                mlp_ratio=mlp_ratio,
            ),
        )

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:  # noqa: ARG002
        """Rollout the processor over a window of history/forecast timesteps."""

        # Get the initial shape of the input tensor
        n, t, c, h, w = x.shape

        # Fold history timesteps into the channel dimension
        x_nchw = x.view(n, t * c, h, w)

        # Apply the translator to get the forecast timesteps
        prediction_nchw: TensorNCHW = self.translator(x_nchw)

        # Unfold the forecast timesteps from the channel dimension
        prediction = prediction_nchw.view(n, -1, c, h, w)

        return ProcessorOutput(prediction=prediction)
