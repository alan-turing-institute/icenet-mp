from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from icenet_mp.models.diffusion import GaussianDiffusion, UNetDiffusion
from icenet_mp.types import BetaSchedule, ProcessorOutput, TensorNCHW, TensorNTCHW

from .base_processor import BaseProcessor


class DDPMProcessor(BaseProcessor):

    def __init__(  # noqa: PLR0913
            self,
            *,
            timesteps: int = 1000,
            beta_schedule: str = "cosine",
            kernel_size: int = 3,
            start_out_channels: int = 64,
            time_embed_dim: int = 256,
            dropout_rate: float = 0.1,
            normalization: str = "groupnorm",
            activation: str = "SiLU",
            use_autoregressive: bool = True,
            target_slice_start: int = 0,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
    
            c_combined = self.data_space.channels
    
            c_target = self.data_space_target.channels
    
            if target_slice_start < 0 or target_slice_start + c_target > c_combined:
                msg = (
                    f"target_slice_start={target_slice_start} with target channels="
                    f"{c_target} does not fit inside combined latent channels="
                    f"{c_combined}."
                )
                raise ValueError(msg)
    
            self.timesteps = timesteps
            self.use_autoregressive = use_autoregressive
            
            self.target_slice_start = target_slice_start
            self.c_target = c_target
            self.c_combined = c_combined
    
            cond_channels = c_combined * self.n_history_steps
    
            diffused_channels = (
                c_target if use_autoregressive else c_target * self.n_forecast_steps
            )
            self.diffused_channels = diffused_channels
    
            self.model = UNetDiffusion(
                input_channels=cond_channels,
                output_channels=diffused_channels,
                timesteps=timesteps,
                kernel_size=kernel_size,
                start_out_channels=start_out_channels,
                time_embed_dim=time_embed_dim,
                normalization=normalization,
                activation=activation,
                dropout_rate=dropout_rate,
            )
            self.diffusion = GaussianDiffusion(
                timesteps=timesteps,
                beta_schedule=BetaSchedule(beta_schedule),
            )

    def rollout(
            self,
            x: TensorNTCHW,
            y: TensorNTCHW | None = None,
        ) -> ProcessorOutput:
            if y is not None:
                return self._training_rollout(x, y)
            return self._inference_rollout(x)       

    def _training_rollout(
            self, x: TensorNTCHW, y: TensorNTCHW
        ) -> ProcessorOutput:
            raise NotImplementedError("Training rollout not yet implemented")

    def _inference_rollout(self, x: TensorNTCHW) -> ProcessorOutput:
            if self.use_autoregressive:
                return ProcessorOutput(prediction=self._sample_autoregressive(x))
            return ProcessorOutput(prediction=self._sample_parallel(x))

    def _sample_parallel(self, x: TensorNTCHW) -> TensorNTCHW:
        raise NotImplementedError("Parallel sampling not yet implemented")

    def _sample_autoregressive(self, x: TensorNTCHW) -> TensorNTCHW:
        raise NotImplementedError("Autoregressive sampling not yet implemented")