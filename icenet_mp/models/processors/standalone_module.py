from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig
from torch import nn

from icenet_mp.types import TensorNCHW

from .base_processor import BaseProcessor


class StandaloneModuleProcessor(BaseProcessor):
    """Adapt a standalone ``nn.Module`` to the IceNet-MP processor interface.

    The wrapped module receives the standard processor input for one forecast step:
    the current history window concatenated along the channel dimension. It must return
    one NCHW latent frame with the same channel count and spatial shape as ``data_space``.

    This keeps standalone model implementations independent of IceNet-MP while allowing
    them to use the existing autoregressive rollout, training and evaluation pipeline.
    """

    def __init__(
        self,
        *,
        module: nn.Module | DictConfig,
        **kwargs: Any,
    ) -> None:
        """Initialise the adapter and wrapped standalone module."""
        super().__init__(**kwargs)
        self.module = (
            module if isinstance(module, nn.Module) else hydra.utils.instantiate(module)
        )
        if not isinstance(self.module, nn.Module):
            msg = (
                "StandaloneModuleProcessor module must instantiate to torch.nn.Module, "
                f"got {type(self.module).__name__}."
            )
            raise TypeError(msg)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Run the standalone module and validate the processor tensor contract."""
        output = self.module(x)
        if not isinstance(output, torch.Tensor):
            msg = (
                "Standalone module must return a torch.Tensor, "
                f"got {type(output).__name__}."
            )
            raise TypeError(msg)

        expected_shape = (x.shape[0], *self.data_space.chw)
        if tuple(output.shape) != expected_shape:
            msg = (
                f"Standalone module returned shape {tuple(output.shape)}, expected "
                f"{expected_shape}."
            )
            raise ValueError(msg)
        return cast("TensorNCHW", output)
