import copy
import logging
from typing import TYPE_CHECKING, Any

import hydra
import torch
from omegaconf import DictConfig

from icenet_mp.models import BaseModel
from icenet_mp.types import ModelStepOutput, TensorNTCHW

from .decoder_stage import DecoderStage

if TYPE_CHECKING:
    from icenet_mp.models.encoders import BaseEncoder
    from icenet_mp.models.processors import BaseProcessor

logger = logging.getLogger(__name__)


class ProcessorStage(BaseModel):
    def __init__(
        self,
        processor: DictConfig,
        decoder_model: DecoderStage,
        **kwargs: Any,
    ) -> None:
        """Initialise a ProcessorStage with frozen encoders, a frozen decoder, and a trainable processor."""
        super().__init__(**kwargs)

        # Copy encoders from DecoderStage, freeze their parameters and register them
        self.encoder_names = decoder_model.encoder_names
        self.encoders = [copy.deepcopy(encoder) for encoder in decoder_model.encoders]
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = False
            self.add_module(encoder.name, encoder)

        # Identify which encoder to use to encode the target if needed
        try:
            target_encoder_idx = self.encoder_names.index(decoder_model.target_name)
        except ValueError:
            msg = (
                f"Target dataset '{decoder_model.target_name}' has no corresponding "
                f"encoder in {self.encoder_names}. ProcessorStage requires an "
                "appropriate encoder for the target dataset to support latent-space "
                "losses."
            )
            raise ValueError(msg) from None
        self.target_encoder: BaseEncoder = self.encoders[target_encoder_idx]

        # Copy combined latent space from DecoderStage
        combined_latent_space = decoder_model.decoder.data_space_in

        # Copy decoder from DecoderStage and freeze it
        self.decoder = copy.deepcopy(decoder_model.decoder)
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
        decoder_model: DecoderStage,
    ) -> "ProcessorStage":
        """Create a ProcessorStage from a trained DecoderStage."""
        return cls(
            processor=processor,
            decoder_model=decoder_model,
            hemisphere=decoder_model.hemisphere,
            input_spaces=[s.to_dict() for s in decoder_model.input_spaces],
            n_forecast_steps=decoder_model.n_forecast_steps,
            n_history_steps=decoder_model.n_history_steps,
            name=decoder_model.name.replace("decoder_model", "processor_model"),
            optimizer=copy.deepcopy(decoder_model.optimizer_cfg),
            output_space=decoder_model.output_space.to_dict(),
            scheduler=copy.deepcopy(decoder_model.scheduler_cfg),
            loss=copy.deepcopy(decoder_model.loss_cfg),
        )

    def encode_inputs(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Encode all input datasets and concatenate along the channel dimension."""
        latent_inputs: list[TensorNTCHW] = [
            encoder.rollout(inputs[encoder.name]) for encoder in self.encoders
        ]
        return torch.cat(latent_inputs, dim=2)

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model (used for inference and the standard decode path).

        - encode each input with frozen encoder.rollout() -> NTCHW latents
        - concatenate latents along the channel dimension
        - process in latent space with trainable processor.rollout() -> NTCHW
        - decode with frozen decoder.rollout() -> output space NTCHW
        """
        combined_latent: TensorNTCHW = self.encode_inputs(inputs)
        return self.decoder.rollout(self.processor.rollout(combined_latent).prediction)

    def training_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> ModelStepOutput:
        """Run the training step.

        If the processor implements compute_loss, both inputs and target are encoded into
        latent space and the processor's custom loss is used for backpropagation. The
        decoded prediction is still computed (under no_grad) so that metrics and callbacks
        remain meaningful.

        Otherwise, the standard encode-process-decode path is used and the loss is
        computed by comparing the decoded prediction to the target.

        Args:
            batch: Dictionary with one NTCHW entry per input dataset (n_history_steps)
                   and a "target" entry (n_forecast_steps).

        Returns:
            A ModelStepOutput containing the prediction, target and loss.

        """
        target = batch["target"].clone().detach()
        combined_latent = self.encode_inputs(batch)
        target_latent = self.target_encoder.rollout(target)

        processor_output = self.processor.rollout(combined_latent, target_latent)

        if processor_output.loss is not None:
            # Custom loss path: processor owns the training signal.
            # Decode under no_grad for metrics/callbacks only.
            loss = processor_output.loss
            with torch.no_grad():
                prediction = self.decoder.rollout(processor_output.prediction)
        else:
            # Standard path: compare decoded output to target.
            prediction = self.decoder.rollout(processor_output.prediction)
            loss = self.loss(prediction, target)

        # Log metrics; computation will be done at epoch end
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_metrics.update(prediction, target)

        return ModelStepOutput(prediction, target, loss)
