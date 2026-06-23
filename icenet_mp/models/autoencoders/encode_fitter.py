import copy
import logging
from typing import TYPE_CHECKING, Any

import hydra
from omegaconf import DictConfig

from icenet_mp.models import BaseModel, EncodeProcessDecode
from icenet_mp.types import ModelStepOutput, TensorNTCHW

if TYPE_CHECKING:
    from icenet_mp.models.decoders import BaseDecoder
    from icenet_mp.models.encoders import BaseEncoder

logger = logging.getLogger(__name__)


class EncodeFitter(BaseModel):
    def __init__(
        self,
        channel_names: list[str],
        dataset: str,
        encoder: DictConfig,
        decoder: DictConfig,
        latent_space: tuple[int, int],
        **kwargs: Any,
    ) -> None:
        """Initialise an EncodeFitter with a trainable encoder and a disposable decoder."""
        super().__init__(**kwargs)

        # Store channel names
        self.channel_names = channel_names

        # Encode from a single input space to a latent space
        self.encoder: BaseEncoder = hydra.utils.instantiate(
            encoder,
            data_space_in=next(s for s in self.input_spaces if s.name == dataset),
            latent_space=latent_space,
            latitudes_fn=self.latitudes_fn,
            longitudes_fn=self.longitudes_fn,
        )

        # Decode from the latent space back to the original input space
        self.decoder: BaseDecoder = hydra.utils.instantiate(
            decoder,
            data_space_in=self.encoder.data_space_out,
            data_space_out=self.encoder.data_space_in,
        )

    @property
    def dataset_name(self) -> str:
        """Return the name of the dataset that this model is fitting."""
        return self.encoder.data_space_in.name

    @classmethod
    def from_template(
        cls,
        *,
        channel_names: list[str],
        dataset: str,
        decoder: DictConfig,
        encoder: DictConfig,
        template: EncodeProcessDecode,
    ) -> "EncodeFitter":
        """Create an EncodeFitter from an existing EncodeProcessDecode template."""
        return cls(
            channel_names=channel_names,
            dataset=dataset,
            decoder=decoder,
            encoder=encoder,
            hemisphere=template.hemisphere,
            input_spaces=[s.to_dict() for s in template.input_spaces],
            latent_space=template.encoders[0].data_space_out.shape,
            n_forecast_steps=template.n_forecast_steps,
            n_history_steps=template.n_history_steps,
            name=f"{template.name}_encoder_fitter",
            optimizer=copy.deepcopy(template.optimizer_cfg),
            output_space=template.output_space.to_dict(),
            scheduler=copy.deepcopy(template.scheduler_cfg),
            loss=copy.deepcopy(template.loss_cfg),
        )

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        - squeeze time dimension to get [NCHW] [batch, n_input_channels, H_input, W_input]
        - encode into latent space [NCHW] [batch, n_latent_channels_total, H_latent, W_latent]
        - decode back to [NCHW] output space [batch, n_output_channels, H_output, W_output]
        - unsqueeze time dimension to get back to [NTCHW] [batch, 1, n_output_channels, H_output, W_output]
        """
        return self.decoder(self.encoder(inputs["target"].squeeze(1))).unsqueeze(1)

    def process_batch(self, batch: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Select only the first time step of only the relevant batch element.

        This is because we want the autoencoder to learn an NCHW -> NCHW mapping and to
        ensure that each input date is only used once per epoch.
        """
        return batch[self.dataset_name][:, 0, :, :, :].unsqueeze(1)

    def training_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> ModelStepOutput:
        """Run the training step.

        A batch contains one tensor for each input dataset and one for the target
        These are [NTCHW] tensors with (batch_size, n_history_steps, C, H, W)

        - Identify the target and take its first time step
        - Pass this through the model to get a prediction
        - Calculate the loss wrt. the target

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A Tensor containing the loss for the batch.

        """
        target = self.process_batch(batch)
        prediction: TensorNTCHW = self({"target": target})
        loss = self.loss(prediction, target)
        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return ModelStepOutput(prediction, target, loss)

    def validation_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> ModelStepOutput:
        """Run the validation step.

        A batch contains one tensor for each input dataset and one for the target
        These are [NTCHW] tensors with (batch_size, n_history_steps, C, H, W)

        - Identify the target and take its first time step
        - Pass this through the model to get a prediction
        - Calculate and log the loss wrt. the target

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A Tensor containing the loss for the batch.

        """
        target = self.process_batch(batch)
        prediction: TensorNTCHW = self({"target": target})
        loss = self.loss(prediction, target)
        self.log(
            "validation_loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return ModelStepOutput(prediction, target, loss)
