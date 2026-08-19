import copy
import logging
from typing import TYPE_CHECKING, Any, ClassVar

import hydra
from omegaconf import DictConfig
from typing_extensions import override

from icenet_mp.models import BaseModel, EncodeProcessDecode

from .decoder_stage import DecoderStage
from .encoder_stage import EncoderStage

if TYPE_CHECKING:
    from icenet_mp.models.processors import BaseProcessor

logger = logging.getLogger(__name__)


class ProcessorStage(EncodeProcessDecode):
    # Parameters that should be excluded from hyperparameter logging
    ignored_hparams: ClassVar[frozenset[str]] = EncodeProcessDecode.ignored_hparams | {
        "decoder_model",
        "target_encoder",
    }

    def __init__(
        self,
        processor: DictConfig,
        decoder_model: DecoderStage,
        target_encoder: EncoderStage,
        **kwargs: Any,
    ) -> None:
        """Initialise a ProcessorStage with frozen encoders, a frozen decoder, and a trainable processor."""
        # We skip EncodeProcessDecode initialisation since we want to use pre-trained
        # encoders, processor and decoder. This relies on the assumption that nothing
        # else is done during initialisation aside from creating these modules.
        BaseModel.__init__(self, **kwargs)

        # Copy encoders from DecoderStage, freeze their parameters and register them.
        self.encoder_names = decoder_model.encoder_names
        self.encoders = [
            copy.deepcopy(encoder).freeze() for encoder in decoder_model.encoders
        ]
        for encoder in self.encoders:
            self.add_module(encoder.name, encoder)

        # Load the target encoder and freeze it
        self.target_encoder = target_encoder.encoder.freeze()

        # Verify the output channels for each encoder
        for encoder in (*self.encoders, self.target_encoder):
            encoder.verify_output_channels(self.device)

        # Copy combined latent space from DecoderStage
        combined_latent_space = decoder_model.decoder.data_space_in

        # Copy decoder from DecoderStage and freeze it
        self.decoder = copy.deepcopy(decoder_model.decoder).freeze()
        self.target_variable_indices = decoder_model.target_variable_indices

        # Trainable processor
        self.processor: BaseProcessor = hydra.utils.instantiate(
            processor,
            data_space=combined_latent_space,
            data_space_target=self.target_encoder.data_space_out,
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
        )

    @classmethod
    def from_template(
        cls,
        *,
        processor: DictConfig,
        decoder_model: DecoderStage,
        target_encoder: EncoderStage,
    ) -> "ProcessorStage":
        """Create a ProcessorStage from a trained DecoderStage."""
        return cls(
            decoder_model=decoder_model,
            hemisphere=decoder_model.hemisphere,
            input_spaces=[s.to_dict() for s in decoder_model.input_spaces],
            loss=copy.deepcopy(decoder_model.loss_cfg),
            n_forecast_steps=decoder_model.n_forecast_steps,
            n_history_steps=decoder_model.n_history_steps,
            name=f"processor_{decoder_model.n_history_steps}_to_{decoder_model.n_forecast_steps}",
            optimizer=copy.deepcopy(decoder_model.optimizer_cfg),
            output_space=decoder_model.output_space.to_dict(),
            processor=processor,
            scheduler=copy.deepcopy(decoder_model.scheduler_cfg),
            target_encoder=target_encoder,
        )

    @override
    def train(self, mode: bool = True) -> "ProcessorStage":
        """Set training mode, but keep frozen modules in eval mode."""
        super().train(mode)
        if mode:
            for encoder in self.encoders:
                encoder.eval()
            self.target_encoder.eval()
            self.decoder.eval()
        return self
