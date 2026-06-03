import copy
import logging
from typing import TYPE_CHECKING, Any

import hydra
import torch
from omegaconf import DictConfig

from icenet_mp.models import BaseModel
from icenet_mp.models.autoencoders.decoder_fitter import DecoderFitter
from icenet_mp.types import TensorNTCHW

if TYPE_CHECKING:
    from icenet_mp.models.processors import BaseProcessor

logger = logging.getLogger(__name__)


class ProcessorFitter(BaseModel):
    def __init__(
        self,
        processor: DictConfig,
        decoder_fitter: DecoderFitter,
        **kwargs: Any,
    ) -> None:
        """Initialise a ProcessorFitter with frozen encoders, a frozen decoder, and a trainable processor."""
        super().__init__(**kwargs)

        # Copy encoders from DecoderFitter, freeze their parameters and register them
        self.encoder_names = decoder_fitter.encoder_names
        self.encoders = [copy.deepcopy(encoder) for encoder in decoder_fitter.encoders]
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = False
            self.add_module(encoder.name, encoder)

        # Copy combined latent space from DecoderFitter
        combined_latent_space = decoder_fitter.decoder.data_space_in

        # Copy decoder from DecoderFitter and freeze it
        self.decoder = copy.deepcopy(decoder_fitter.decoder)
        for param in self.decoder.parameters():
            param.requires_grad = False

        # Trainable processor
        self.processor: BaseProcessor = hydra.utils.instantiate(
            processor,
            data_space=combined_latent_space,
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
        )

    @classmethod
    def from_template(
        cls,
        *,
        processor: DictConfig,
        decoder_fitter: DecoderFitter,
    ) -> "ProcessorFitter":
        """Create a ProcessorFitter from a trained DecoderFitter."""
        return cls(
            processor=processor,
            decoder_fitter=decoder_fitter,
            hemisphere=decoder_fitter.hemisphere,
            input_spaces=[s.to_dict() for s in decoder_fitter.input_spaces],
            n_forecast_steps=decoder_fitter.n_forecast_steps,
            n_history_steps=decoder_fitter.n_history_steps,
            name=decoder_fitter.name.replace("decoder_fitter", "processor_fitter"),
            optimizer=copy.deepcopy(decoder_fitter.optimizer_cfg),
            output_space=decoder_fitter.output_space.to_dict(),
            scheduler=copy.deepcopy(decoder_fitter.scheduler_cfg),
        )

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        - encode each input with frozen encoder.rollout() -> NTCHW latents
        - concatenate latents along the channel dimension
        - process in latent space with trainable processor.rollout() -> NTCHW
        - decode with frozen decoder.rollout() -> output space NTCHW
        """
        latent_inputs: list[TensorNTCHW] = [
            encoder.rollout(inputs[encoder.name]) for encoder in self.encoders
        ]
        latent_input_combined: TensorNTCHW = torch.cat(latent_inputs, dim=2)
        latent_output: TensorNTCHW = self.processor.rollout(
            latent_input_combined, inputs.get("target")
        )
        return self.decoder.rollout(latent_output)
