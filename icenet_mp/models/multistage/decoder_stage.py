import copy
import logging
from typing import TYPE_CHECKING, Any

import hydra
import torch
from omegaconf import DictConfig

from icenet_mp.models import BaseModel
from icenet_mp.types import DataSpace, TensorNTCHW

from .encoder_stage import EncoderStage

if TYPE_CHECKING:
    from icenet_mp.models.decoders import BaseDecoder

logger = logging.getLogger(__name__)


class DecoderStage(BaseModel):
    def __init__(
        self,
        decoder: DictConfig,
        encoders: list[EncoderStage],
        target_dataset_name: str,
        target_variable_indices: list[int],
        active_mask_path: str | None = None,
        land_mask_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise a DecoderStage with multiple frozen encoders and a trainable decoder."""
        super().__init__(**kwargs)
        self.active_mask_path = active_mask_path
        self.land_mask_path = land_mask_path

        # Copy encoders from EncoderStages, freeze their parameters and register them
        self.encoder_names = [encoder.dataset_name for encoder in encoders]
        self.encoders = [copy.deepcopy(encoder.encoder) for encoder in encoders]
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = False
            self.add_module(encoder.name, encoder)

        # Build combined latent DataSpace by summing channels across all encoders
        total_channels = sum(
            encoder.data_space_out.channels for encoder in self.encoders
        )
        combined_latent_space = DataSpace(
            channels=total_channels,
            name="combined_latent",
            shape=self.encoders[0].data_space_out.shape,
        )

        # Decode from combined latent space to the configured output space
        self.target_name = target_dataset_name
        self.target_indices = target_variable_indices
        if self.output_space.channels != len(target_variable_indices):
            msg = (
                f"output_space has {self.output_space.channels} channel(s) but "
                f"target_variable_indices selects {len(target_variable_indices)}; "
                f"check that predict.target.variables is set correctly."
            )
            raise ValueError(msg)
        self.decoder: BaseDecoder = hydra.utils.instantiate(
            decoder,
            active_mask_path=self.active_mask_path,
            data_space_in=combined_latent_space,
            data_space_out=self.output_space,
            land_mask_path=self.land_mask_path,
        )

    @classmethod
    def from_template(
        cls,
        *,
        decoder: DictConfig,
        encoders: list[EncoderStage],
        target_dataset_name: str,
        target_variable_indices: list[int],
        active_mask_path: str | None = None,
        land_mask_path: str | None = None,
    ) -> "DecoderStage":
        """Create a DecoderStage from a list of trained EncoderStages."""
        return cls(
            decoder=decoder,
            encoders=encoders,
            target_dataset_name=target_dataset_name,
            target_variable_indices=target_variable_indices,
            active_mask_path=active_mask_path,
            land_mask_path=land_mask_path,
            hemisphere=encoders[0].hemisphere,
            input_spaces=[s.to_dict() for s in encoders[0].input_spaces],
            n_forecast_steps=encoders[0].n_forecast_steps,
            n_history_steps=encoders[0].n_history_steps,
            name=f"{target_dataset_name}_decoder".replace("-", "_"),
            optimizer=copy.deepcopy(encoders[0].optimizer_cfg),
            output_space=encoders[0].output_space.to_dict(),
            scheduler=copy.deepcopy(encoders[0].scheduler_cfg),
            loss=copy.deepcopy(encoders[0].loss_cfg),
        )

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        - squeeze time dimension from each input to get [NCHW]
        - encode each dataset with its corresponding frozen encoder
        - concatenate latent representations along the channel dimension
        - decode to target space [NCHW]
        - unsqueeze time dimension to return [NTCHW]
        """
        latents = [
            encoder(inputs[name].squeeze(1))
            for name, encoder in zip(self.encoder_names, self.encoders, strict=True)
        ]
        combined = torch.cat(latents, dim=1)
        return self.decoder(combined).unsqueeze(1)

    def process_batch(
        self,
        batch: dict[str, TensorNTCHW],
    ) -> dict[str, TensorNTCHW]:
        """Extract only the first time step of each relevant batch element.

        This is because we want the decoder to learn an NCHW -> NCHW mapping and to
        ensure that each input date is only used once per epoch.
        """
        return {
            name: batch[name][:, 0, :, :, :].unsqueeze(1) for name in self.encoder_names
        } | {
            "target": batch[self.target_name][
                :, 0, self.target_indices, :, :
            ].unsqueeze(1)
        }
