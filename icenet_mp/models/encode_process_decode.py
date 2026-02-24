from typing import TYPE_CHECKING, Any

import hydra
import torch
from omegaconf import DictConfig

from icenet_mp.types import DataSpace, TensorNTCHW

from .base_model import BaseModel

if TYPE_CHECKING:
    from icenet_mp.models.decoders import BaseDecoder
    from icenet_mp.models.encoders import BaseEncoder
    from icenet_mp.models.processors import BaseProcessor


class EncodeProcessDecode(BaseModel):
    def __init__(
        self,
        *,
        encoder: DictConfig,
        processor: DictConfig,
        decoder: DictConfig,
        **kwargs: Any,
    ) -> None:
        """Initialise an EncodeProcessDecode model."""
        super().__init__(**kwargs)

        # Add one encoder per dataset
        # We store this as a list to ensure consistent ordering
        self.encoders: list[BaseEncoder] = [
            hydra.utils.instantiate(
                dict(**encoder)
                | {
                    "data_space_in": input_space,
                    "n_history_steps": self.n_history_steps,
                }
            )
            for input_space in self.input_spaces
        ]

        # We have to explicitly register each encoder as list[Module] will not be
        # automatically picked up by PyTorch
        for input_space, module in zip(self.input_spaces, self.encoders, strict=True):
            module_name = f"encoder_{input_space.name}".lower().replace("-", "_")
            self.add_module(module_name, module)

        # Add a processor
        combined_latent_space = DataSpace(
            name="combined_latent_space",
            channels=sum(encoder.data_space_out.channels for encoder in self.encoders),
            shape=self.encoders[0].data_space_out.shape,
        )
        self.processor: BaseProcessor = hydra.utils.instantiate(
            dict(**processor)
            | {
                "data_space": combined_latent_space,
                "n_forecast_steps": self.n_forecast_steps,
                "n_history_steps": self.n_history_steps,
            }
        )

        # Add a decoder
        self.decoder: BaseDecoder = hydra.utils.instantiate(
            dict(**decoder)
            | {
                "data_space_in": combined_latent_space,
                "data_space_out": self.output_space,
                "n_forecast_steps": self.n_forecast_steps,
            }
        )

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        - start with multiple [NTCHW] inputs each with shape [batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k]
        - encode inputs to [NTCHW] latent space [batch, n_history_steps, n_latent_channels, H_latent, W_latent]
        - concatenate inputs in [NTCHW] latent space [batch, n_history_steps, n_latent_channels_total, H_latent, W_latent]
        - process in latent space [NTCHW] [batch, n_forecast_steps, n_latent_channels_total, H_latent, W_latent]
        - decode back to [NTCHW] output space [batch, n_forecast_steps, n_output_channels, H_output, W_output]
        """
        # Encode inputs into latent space: list of tensors with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)
        latent_inputs: list[TensorNTCHW] = [
            encoder.rollout(inputs[encoder.name]) for encoder in self.encoders
        ]

        # Combine in the variable dimension: tensor with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
        latent_input_combined: TensorNTCHW = torch.cat(latent_inputs, dim=2)

        # Process in latent space:
        # combined input tensor with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
        # target tensor with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width) or None
        latent_output: TensorNTCHW = self.processor.rollout(
            latent_input_combined, inputs.get("target")
        )

        # Decode to output space: tensor with (batch_size, n_forecast_steps, n_output_channels, output_height, output_width)
        output: TensorNTCHW = self.decoder.rollout(latent_output)

        # Return
        return output
