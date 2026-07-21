"""Processor that approximates SimVPv2.

Reference:
    SimVP: Simpler yet Better Video Prediction
    (Gao et al., 2022) [https://arxiv.org/abs/2206.05099]
    SimVPv2: Towards Simple yet Powerful Spatiotemporal Predictive Learning
    (Tan et al., 2022) [https://arxiv.org/abs/2211.12509]

SimVP is a spatio-temporal predictor made of three plain-CNN pieces:
    1. A per-timestep spatial encoder built from ``ConvBlockDownsample`` blocks.
    2. A temporal translator: a flat stack of "gSTA" (Gated Spatiotemporal Attention)
       blocks replacing the original SimVP Inception-based U-Net structure.
    3. A per-timestep spatial decoder built from ``ConvBlockUpsample`` blocks that
       mirrors the encoder.

Note that we use slightly different encoder/decoder blocks and also adapt the translator
and decoder to allow for different numbers of history and forecast steps.
"""

from typing import Any

from torch import cat, nn

from icenet_mp.models.common import (
    ConvBlockDownsample,
    ConvBlockUpsample,
    GatedAttentionBlock,
)
from icenet_mp.types import ProcessorOutput, TensorNTCHW

from .base_processor import BaseProcessor


class _SpatialEncoder(nn.Module):
    """Per-frame spatial encoder: one `ConvBlockDownsample` per stride-1/stride-2 step.

    Mirror of :class:`_SpatialDecoder`.

    The paper's alternating stride-1/stride-2 pattern always starts with a stride-1,
    channel-projecting layer at full resolution (no downsampling) -- this first layer
    is what the decoder's skip connection is taken from.
    """

    def __init__(self, in_channels: int, hid_channels: int, depth: int) -> None:
        super().__init__()
        strides = ([1, 2] * depth)[:depth]
        block_kwargs = {
            "activation": "LeakyReLU",
            "kernel_size": 3,
            "n_subblocks": 1,
            "norm_type": "groupnorm",
            "out_channels": hid_channels,
        }
        self.channel_conv = ConvBlockDownsample(
            in_channels, scale_factor=strides[0], **block_kwargs
        )
        self.downsample_convs = nn.ModuleList(
            ConvBlockDownsample(hid_channels, scale_factor=stride, **block_kwargs)
            for stride in strides[1:]
        )

    def forward(self, x: TensorNTCHW) -> tuple[TensorNTCHW, TensorNTCHW]:
        """Return (final latent, first-layer activations) for the decoder skip connection."""
        n, t, c, h, w = x.shape
        skip_nchw = self.channel_conv(x.view(n * t, c, h, w))
        z = skip_nchw
        for downsample_conv in self.downsample_convs:
            z = downsample_conv(z)
        return z.view(n, t, *z.shape[1:]), skip_nchw.view(n, t, *skip_nchw.shape[1:])


class _SpatialDecoder(nn.Module):
    """Per-frame spatial decoder: one `ConvBlockUpsample` per stride-1/stride-2 step.

    Mirror of :class:`_SpatialEncoder`.

    As in `_SpatialEncoder`, the paper's alternating pattern reversed always ends with
    stride 1, so the final layer is always a single channel-projecting layer at full
    resolution -- the one that receives the concatenated encoder skip connection.
    """

    def __init__(self, hid_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        strides = list(reversed(([1, 2] * depth)[:depth]))
        block_kwargs = {
            "activation": "LeakyReLU",
            "kernel_size": 3,
            "n_subblocks": 1,
            "norm_type": "groupnorm",
            "out_channels": hid_channels,
            "upsample_mode": "shuffle",
        }
        self.upsample_convs = nn.ModuleList(
            ConvBlockUpsample(hid_channels, scale_factor=stride, **block_kwargs)
            for stride in strides[:-1]
        )
        self.skip_conv = ConvBlockUpsample(
            2 * hid_channels, scale_factor=strides[-1], **block_kwargs
        )
        self.channel_conv = nn.Conv2d(hid_channels, out_channels, kernel_size=1)

    def forward(self, z: TensorNTCHW, skip: TensorNTCHW) -> TensorNTCHW:
        # z has shape [N, n_forecast_steps, C_hid, H_hid, W_hid]
        # skip has shape [N, n_forecast_steps, C_hid, H, W]
        n, t, c, h, w = z.shape
        z_nchw = z.view(n * t, c, h, w)
        skip_nchw = skip.reshape(n * t, c, *skip.shape[3:])
        for upsample_conv in self.upsample_convs:
            z_nchw = upsample_conv(z_nchw)
        z_nchw = self.skip_conv(cat([z_nchw, skip_nchw], dim=1))
        output = self.channel_conv(z_nchw)
        return output.view(n, t, *output.shape[1:])


class SimVPProcessor(BaseProcessor):
    """Processor that approximates SimVPv2.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dilation: int = 3,
        drop_prob: float = 0.1,
        hid_channels_spatial: int = 64,
        hid_channels_temporal: int = 256,
        kernel_size: int = 21,
        mlp_ratio: float = 4.0,
        spatial_depth: int = 4,
        temporal_depth: int = 8,
        **kwargs: Any,
    ) -> None:
        """Initialise a SimVPProcessor.

        Args:
            dilation: Dilation used by gated attention blocks in the translator.
            drop_prob: Stochastic depth drop probability for residual branches in the translator.
            hid_channels_spatial: Hidden channel count for the spatial encoder/decoder (hid_S).
            hid_channels_temporal: Hidden channel count for the temporal translator (hid_T).
            kernel_size: Effective receptive field of the translator's gated attention blocks.
            mlp_ratio: Hidden-channel expansion ratio for the translator's conv-MLP blocks.
            spatial_depth: Number of ConvBlockDownsample/ConvBlockUpsample layers in the spatial encoder/decoder (N_S).
            temporal_depth: Number of gSTA blocks in the temporal translator (N_T).
            kwargs: Arguments to BaseProcessor.

        """
        super().__init__(**kwargs)

        if spatial_depth < 1:
            msg = f"spatial_depth must be at least 1, got {spatial_depth}"
            raise ValueError(msg)
        if temporal_depth < 2:  # noqa: PLR2004
            msg = f"temporal_depth must be at least 2 to project in_channels -> hid_channels -> out_channels, got {temporal_depth}"
            raise ValueError(msg)

        # The encoder/decoder alternate stride-1 and stride-2 layers, so this is the
        # total spatial down/upsampling factor applied by the encoder/decoder.
        downsample_factor = 2 ** (spatial_depth // 2)
        height, width = self.data_space.shape
        if height % downsample_factor or width % downsample_factor:
            msg = (
                f"Latent space height ({height}) and width ({width}) must each be "
                f"divisible by the encoder's downsampling factor ({downsample_factor}, "
                f"determined by spatial_depth={spatial_depth})."
            )
            raise ValueError(msg)

        self.encoder = _SpatialEncoder(
            self.data_space.channels, hid_channels_spatial, spatial_depth
        )

        # Translator: a flat stack of gSTA blocks, with time folded into the channels
        translator_kwargs = {
            "kernel_size": kernel_size,
            "dilation": dilation,
            "mlp_ratio": mlp_ratio,
            "drop_prob": drop_prob,
        }
        self.translator = nn.Sequential(
            GatedAttentionBlock(
                self.n_history_steps * hid_channels_spatial,
                hid_channels_temporal,
                **translator_kwargs,
            ),
            *(
                GatedAttentionBlock(
                    hid_channels_temporal, hid_channels_temporal, **translator_kwargs
                )
                for _ in range(temporal_depth - 2)
            ),
            GatedAttentionBlock(
                hid_channels_temporal,
                self.n_forecast_steps * hid_channels_spatial,
                **translator_kwargs,
            ),
        )
        self.decoder = _SpatialDecoder(
            hid_channels_spatial, self.data_space.channels, spatial_depth
        )

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:  # noqa: ARG002
        if x.shape[1] != self.n_history_steps:
            msg = f"Expected T={self.n_history_steps}, got {x.shape[1]}"
            raise ValueError(msg)

        # Spatial encoder: applied independently per-timeslice.
        # -> [N, n_history_steps, C_hid, H_hid, W_hid]
        embed, skip_history = self.encoder(x)

        # Temporal translator: stack timeslices along the channel dimension, apply the
        # gSTA blocks, then unstack -> [N, n_forecast_steps, C_hid, H_hid, W_hid]
        n, t, c, h, w = embed.shape
        z = self.translator(embed.view(n, t * c, h, w)).view(n, -1, c, h, w)

        # The skip connection needs exactly n_forecast_steps timeslices.
        # - n_history_steps > n_forecast_steps: use the most recent n_forecast_steps
        # - n_history_steps = n_forecast_steps: use all steps
        # - n_history_steps < n_forecast_steps: use all available steps, then repeat the most recent one
        # -> [N, n_forecast_steps, C_hid, H_hid, W_hid]
        skip_forecast = skip_history[
            :,
            [
                min(
                    max(0, self.n_history_steps - self.n_forecast_steps) + i,
                    self.n_history_steps - 1,
                )
                for i in range(self.n_forecast_steps)
            ],
        ]

        # Spatial decoder: applied independently per-timeslice.
        # -> [N, n_forecast_steps, C, H, W]
        yhat = self.decoder(z, skip_forecast)

        return ProcessorOutput(prediction=yhat)
