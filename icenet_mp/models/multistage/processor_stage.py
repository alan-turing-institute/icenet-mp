import copy
import logging
from typing import TYPE_CHECKING, Any

import hydra
import torch
from omegaconf import DictConfig
from typing_extensions import override

from icenet_mp.models import BaseModel
from icenet_mp.types import ModelStepOutput, TensorNTCHW

from .decoder_stage import DecoderStage
from .encoder_stage import EncoderStage

if TYPE_CHECKING:
    from icenet_mp.models.processors import BaseProcessor

logger = logging.getLogger(__name__)


class ProcessorStage(BaseModel):
    def __init__(
        self,
        processor: DictConfig,
        decoder_model: DecoderStage,
        target_encoder: EncoderStage,
        **kwargs: Any,
    ) -> None:
        """Initialise a ProcessorStage with frozen encoders, a frozen decoder, and a trainable processor."""
        super().__init__(**kwargs)

        # Copy encoders from DecoderStage, freeze their parameters and register them.
        self.encoder_names = decoder_model.encoder_names
        self.encoders = [
            copy.deepcopy(encoder).freeze() for encoder in decoder_model.encoders
        ]
        for encoder in self.encoders:
            self.add_module(encoder.name, encoder)

        # Load the target encoder and freeze it
        self.target_encoder = target_encoder.encoder.freeze()

        # Copy combined latent space from DecoderStage
        combined_latent_space = decoder_model.decoder.data_space_in

        # Copy decoder from DecoderStage and freeze it
        self.decoder = copy.deepcopy(decoder_model.decoder).freeze()

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

    def training_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> ModelStepOutput:
        """Run the training step.

        If the processor returns a loss in its `ProcessorOutput` (rather than `None`),
        this is used for backpropagation. We use `no_grad` to compute the decoded
        prediction, which allows us to calculate metrics and log outputs, but the
        usefulness of these will depend on what `ProcessorOutput.prediction` contains.

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

        # Attempt to encode the target to latent space and pass it to the processor
        expected_chw = self.target_encoder.data_space_in.chw
        if tuple(target.shape[2:]) != expected_chw:
            msg = (
                f"Target CHW {tuple(target.shape[2:])} does not match "
                f"'{self.target_encoder.name}' encoder input (C, H, W)={expected_chw}."
            )
            raise ValueError(msg)
        target_latent = self.target_encoder.rollout(target)
        processor_output = self.processor.rollout(combined_latent, target_latent)

        if processor_output.loss is None:
            # Standard path: compare decoded output to target.
            prediction = self.decoder.rollout(processor_output.prediction)
            loss = self.loss(prediction, target)
        else:
            # Custom loss path: processor owns the training signal.
            # Decode under no_grad for metrics/callbacks only.
            loss = processor_output.loss
            with torch.no_grad():
                prediction = self.decoder.rollout(processor_output.prediction)

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
