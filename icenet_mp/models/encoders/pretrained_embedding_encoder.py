from typing import Any, Literal

from torch import nn
from torch.nn import functional

from icenet_mp.models.common.normalisations import normalisation_from_name
from icenet_mp.types import DataSpace, TensorNCHW

from .base_encoder import BaseEncoder


class PretrainedEmbeddingEncoder(BaseEncoder):
    """Adapt pre-computed gridded embeddings to the shared latent space.

    This encoder deliberately avoids spatial feature extraction. If the incoming
    embedding already has the requested spatial shape and channel count, the default
    configuration is an exact pass-through. Otherwise it can resize the embedding map
    and/or use a 1x1 convolution to project embedding channels into the latent space.

    This is intended for externally generated environmental embeddings where the
    expensive representation learning has already happened upstream.
    """

    _VALID_INTERPOLATION_MODES = frozenset(("nearest", "bilinear", "bicubic"))

    def __init__(
        self,
        *,
        data_space_in: DataSpace,
        latent_space: tuple[int, int],
        output_channels: int | None = None,
        interpolation_mode: Literal["nearest", "bilinear", "bicubic"] = "bilinear",
        norm_type: str = "none",
        **kwargs: Any,
    ) -> None:
        """Initialise the pretrained-embedding adapter."""
        if output_channels is None:
            output_channels = data_space_in.channels
        if output_channels <= 0:
            msg = "output_channels must be greater than 0."
            raise ValueError(msg)
        if interpolation_mode not in self._VALID_INTERPOLATION_MODES:
            msg = (
                f"Unsupported interpolation_mode {interpolation_mode!r}; expected one "
                f"of {sorted(self._VALID_INTERPOLATION_MODES)}."
            )
            raise ValueError(msg)

        super().__init__(
            data_space_in=data_space_in,
            latent_space=latent_space,
            output_channels=output_channels,
            **kwargs,
        )

        self.interpolation_mode = interpolation_mode
        self.projection: nn.Module = (
            nn.Identity()
            if data_space_in.channels == output_channels
            else nn.Conv2d(data_space_in.channels, output_channels, kernel_size=1)
        )
        self.normalisation = normalisation_from_name(norm_type, output_channels)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Resize/project an embedding map without adding spatial convolutions."""
        if tuple(x.shape[-2:]) != self.data_space_out.shape:
            kwargs: dict[str, bool] = {}
            if self.interpolation_mode in {"bilinear", "bicubic"}:
                kwargs["align_corners"] = False
            x = functional.interpolate(
                x,
                size=self.data_space_out.shape,
                mode=self.interpolation_mode,
                **kwargs,
            )
        return self.normalisation(self.projection(x))
