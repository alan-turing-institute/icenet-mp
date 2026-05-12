from typing import TYPE_CHECKING, Any

import hydra
import torch
from omegaconf import DictConfig

from icenet_mp.types import DataSpace, TensorNTCHW

from .base_model import BaseModel

if TYPE_CHECKING:
    from icenet_mp.models.decoders import BaseDecoder
    from icenet_mp.models.encoders import BaseEncoder


class AutoEncoderModel(BaseModel):
    """Combined autoencoder: all sources encode to a joint latent, each source is reconstructed from it."""

    def __init__(
        self, *, encoders: DictConfig, decoder: DictConfig, **kwargs: Any
    ) -> None:
        """Initialise the AutoEncoder model."""
        super().__init__(**kwargs)

        self.encoders: list[BaseEncoder] = [
            hydra.utils.instantiate(
                encoders[input_space.name],
                data_space_in=input_space,
                latent_space=encoders["latent_space"],
                latitudes_fn=self.latitudes_fn,
                longitudes_fn=self.longitudes_fn,
                n_history_steps=self.n_history_steps,
            )
            for input_space in self.input_spaces
        ]
        for input_space, module in zip(self.input_spaces, self.encoders, strict=True):
            module_name = f"encoder_{input_space.name}".lower().replace("-", "_")
            self.add_module(module_name, module)

        # Construct the combined latent space
        combined_latent_space = DataSpace(
            name="combined_latent_space",
            channels=sum(enc.data_space_out.channels for enc in self.encoders),
            shape=self.encoders[0].data_space_out.shape,
        )

        # Each decoder receives the full combined latent and reconstructs its source
        self.decoders: list[BaseDecoder] = [
            hydra.utils.instantiate(
                decoder,
                data_space_in=combined_latent_space,
                data_space_out=encoder.data_space_in,
                n_forecast_steps=self.n_forecast_steps,
            )
            for encoder in self.encoders
        ]
        for input_space, module in zip(self.input_spaces, self.decoders, strict=True):
            module_name = f"decoder_{input_space.name}".lower().replace("-", "_")
            self.add_module(module_name, module)

    def forward(self, inputs: dict[str, TensorNTCHW]) -> dict[str, TensorNTCHW]:
        latents: list[TensorNTCHW] = [
            encoder(inputs[encoder.name]) for encoder in self.encoders
        ]
        combined: TensorNTCHW = torch.cat(latents, dim=2)
        return {
            encoder.name: decoder(combined)
            for encoder, decoder in zip(self.encoders, self.decoders, strict=True)
        }

    def training_step(
        self, batch: dict[str, TensorNTCHW], _batch_idx: int
    ) -> torch.Tensor:
        inputs = {encoder.name: batch[encoder.name] for encoder in self.encoders}
        loss = self.loss(self(inputs), inputs)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: dict[str, TensorNTCHW], _batch_idx: int
    ) -> torch.Tensor:
        inputs = {encoder.name: batch[encoder.name] for encoder in self.encoders}
        loss = self.loss(self(inputs), inputs)
        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss
