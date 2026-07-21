"""SimVP processor.

Reference:
    SimVP: Simpler yet Better Video Prediction
    (Gao et al., 2022) [https://arxiv.org/abs/2206.05099]

SimVP is a spatio-temporal predictor made of three plain-CNN pieces:
    1. A per-timestep spatial encoder consisting of stacked ConvNormAct blocks
    2. A temporal translator ("Mid-Xnet") using a U-Net-shaped stack of
       ``Inception`` modules (a 1x1 bottleneck followed by parallel grouped
       convolutions at different kernel sizes).
    3. A per-timestep spatial decoder that mirrors the encoder.
"""

from typing import Any

from torch import Tensor, cat, nn

from icenet_mp.models.common import ConvNormAct
from icenet_mp.models.common.activations import ACTIVATION_FROM_NAME
from icenet_mp.models.common.normalisations import normalisation_from_name
from icenet_mp.types import ProcessorOutput, TensorNCHW, TensorNTCHW

from .base_processor import BaseProcessor


class _ConvSC(nn.Module):
    """Spatial conv block: (transposed) Conv2d + GroupNorm + LeakyReLU.

    The non-transposed path is built from the shared `ConvNormAct` block. Transposed
    convolution (used by the decoder to upsample) isn't something `ConvNormAct`
    supports, so that path reuses the `normalisation_from_name`/`ACTIVATION_FROM_NAME`
    helpers directly instead.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        transpose: bool = False,
    ) -> None:
        super().__init__()
        # A stride of 1 is a no-op for up/downsampling, so there is no need to transpose it.
        transpose = transpose and stride != 1
        if transpose:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    output_padding=stride // 2,
                ),
                normalisation_from_name("groupnorm", out_channels),
                ACTIVATION_FROM_NAME["LeakyReLU"](),
            )
        else:
            self.block = ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation="LeakyReLU",
                norm_type="groupnorm",
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _Inception(nn.Module):
    """Inception-style block: 1x1 bottleneck, then parallel grouped convs, summed."""

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        *,
        kernel_sizes: tuple[int, ...],
        groups: int,
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, hid_channels, kernel_size=1)
        # A grouped convolution requires both channel counts to be divisible by groups;
        # fall back to an ungrouped convolution otherwise (matches the reference
        # implementation, which guards this the same way).
        effective_groups = (
            groups if hid_channels % groups == 0 and out_channels % groups == 0 else 1
        )
        self.branches = nn.ModuleList(
            [
                ConvNormAct(
                    hid_channels,
                    out_channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=effective_groups,
                    activation="LeakyReLU",
                    norm_type="groupnorm",
                )
                for k in kernel_sizes
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduce(x)
        y = self.branches[0](x)
        for branch in self.branches[1:]:
            y = y + branch(x)
        return y


class _Encoder(nn.Module):
    """Per-frame spatial encoder: a stack of `_ConvSC` blocks."""

    def __init__(self, in_channels: int, hid_channels: int, depth: int) -> None:
        super().__init__()
        strides = ([1, 2] * depth)[:depth]
        self.channel_conv = _ConvSC(in_channels, hid_channels, stride=strides[0])
        self.layers = nn.ModuleList(
            _ConvSC(hid_channels, hid_channels, stride=s) for s in strides[1:]
        )

    def forward(self, x: TensorNCHW) -> tuple[TensorNCHW, TensorNCHW]:
        """Return (final latent, first-layer activations) for the decoder skip connection."""
        skip = self.channel_conv(x)
        z = skip
        for layer in self.layers:
            z = layer(z)
        return z, skip


class _Decoder(nn.Module):
    """Per-frame spatial decoder: transposed `_ConvSC` blocks mirroring the encoder."""

    def __init__(self, hid_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        strides = list(reversed(([1, 2] * depth)[:depth]))
        self.layers = nn.ModuleList(
            [
                _ConvSC(hid_channels, hid_channels, stride=s, transpose=True)
                for s in strides[:-1]
            ]
        )
        self.skip_conv = _ConvSC(
                    2 * hid_channels, hid_channels, stride=strides[-1], transpose=True
                )
        self.channel_conv = nn.Conv2d(hid_channels, out_channels, kernel_size=1)

    def forward(self, z: TensorNCHW, skip: TensorNCHW) -> TensorNCHW:
        for layer in self.layers[:-1]:
            z = layer(z)
        z = self.skip_conv(cat([z, skip], dim=1))
        return self.channel_conv(z)


class _MidXNet(nn.Module):
    """Temporal translator ("Mid-Xnet"): a U-Net of `_Inception` blocks over stacked frames.

    Input/output are frames stacked along the channel dimension, i.e. (N, T*C, H, W).
    In the original SimVP, in_channels == out_channels (T_in == T_out); here they may
    differ so that a different number of forecast steps than history steps can be
    produced (see the module docstring).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: int,
        depth: int,
        *,
        kernel_sizes: tuple[int, ...],
        groups: int,
    ) -> None:
        super().__init__()
        if depth < 2:  # noqa: PLR2004
            msg = f"temporal_depth must be at least 2 to form the translator's encoder/decoder skip structure, got {depth}"
            raise ValueError(msg)
        self.depth = depth

        def inception(c_in: int, c_out: int) -> _Inception:
            return _Inception(
                c_in, hid_channels // 2, c_out, kernel_sizes=kernel_sizes, groups=groups
            )

        self.enc = nn.ModuleList(
            [inception(in_channels, hid_channels)]
            + [inception(hid_channels, hid_channels) for _ in range(depth - 2)]
            + [inception(hid_channels, hid_channels)]
        )
        self.dec = nn.ModuleList(
            [inception(hid_channels, hid_channels)]
            + [inception(2 * hid_channels, hid_channels) for _ in range(depth - 2)]
            + [inception(2 * hid_channels, out_channels)]
        )

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        z = x
        for i, layer in enumerate(self.enc):
            z = layer(z)
            if i < self.depth - 1:
                skips.append(z)

        z = self.dec[0](z)
        for i in range(1, self.depth):
            z = self.dec[i](cat([z, skips[-i]], dim=1))
        return z


class SimVPProcessor(BaseProcessor):
    """SimVP: Simpler yet Better Video Prediction (Gao et al., 2022).

    See the module docstring for the architecture and for how this differs from the
    original (fixed T_in == T_out) model when `n_history_steps != n_forecast_steps`.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        groups: int = 8,
        hid_channels_spatial: int = 64,
        hid_channels_temporal: int = 256,
        kernel_sizes: tuple[int, ...] = (3, 5, 7, 11),
        spatial_depth: int = 4,
        temporal_depth: int = 8,
        **kwargs: Any,
    ) -> None:
        """Initialise a SimVPProcessor.

        Args:
            groups: Number of groups for the translator's grouped convolutions.
            hid_channels_spatial: Hidden channel count for the spatial encoder/decoder (hid_S).
            hid_channels_temporal: Hidden channel count for the temporal translator (hid_T).
            kernel_sizes: Kernel sizes used by the translator's parallel Inception branches.
            spatial_depth: Number of ConvSC layers in the spatial encoder/decoder (N_S).
            temporal_depth: Number of Inception layers in the temporal translator (N_T).
            kwargs: Arguments to BaseProcessor.

        """
        super().__init__(**kwargs)

        if spatial_depth < 1:
            msg = f"spatial_depth must be at least 1, got {spatial_depth}"
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

        self.encoder = _Encoder(
            self.data_space.channels, hid_channels_spatial, spatial_depth
        )
        self.translator = _MidXNet(
            self.n_history_steps * hid_channels_spatial,
            self.n_forecast_steps * hid_channels_spatial,
            hid_channels_temporal,
            temporal_depth,
            kernel_sizes=kernel_sizes,
            groups=groups,
        )
        self.decoder = _Decoder(
            hid_channels_spatial, self.data_space.channels, spatial_depth
        )

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:  # noqa: ARG002
        n, t, c, h, w = x.shape
        if t != self.n_history_steps:
            msg = f"Expected T={self.n_history_steps}, got {t}"
            raise ValueError(msg)

        # Spatial encoder: applied independently per-timeslice. Both embed and skip
        # come out with shape (N*T, hid_S, H_hid, W_hid).
        embed, skip = self.encoder(x.view(n * t, c, h, w))
        _, c_hid, h_hid, w_hid = embed.shape

        # Move the history timeslices to the channel dimension, apply temporal
        # translation, then move the forecast timeslices back to the time dimension,
        # ending up with shape (N * n_forecast_steps, hid_S, H_hid, W_hid).
        z = self.translator(embed.view(n, t * c_hid, h_hid, w_hid))
        z = z.view(n * self.n_forecast_steps, c_hid, h_hid, w_hid)

        skip = skip.view(n, t, *skip.shape[1:])
        if self.n_forecast_steps == self.n_history_steps:
            # As in the original SimVP: decode forecast frame i using the first
            # encoder layer's activations for that same history frame i.
            skip_dec = skip
        else:
            # The paper only ever predicts as many frames as it is given, so it does
            # not define a skip connection for this case. Reuse the most recent
            # history frame's skip features for every forecast frame instead of
            # leaving the decoder's skip connection undefined.
            skip_dec = skip[:, -1:].expand(-1, self.n_forecast_steps, -1, -1, -1)
        # `.expand()` above (when taken) produces a non-contiguous tensor, so this
        # merge needs `.reshape()`; `.view()` would raise in that branch.
        skip_dec = skip_dec.reshape(n * self.n_forecast_steps, *skip_dec.shape[2:])

        # Spatial decoder: applied per-frame, independently of the other forecast frames.
        yhat2 = self.decoder(z, skip_dec)
        yhat = yhat2.view(n, self.n_forecast_steps, c, h, w)

        return ProcessorOutput(prediction=yhat)
