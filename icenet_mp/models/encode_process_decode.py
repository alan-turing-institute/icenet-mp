from typing import TYPE_CHECKING, Any, ClassVar

import hydra
import torch
from omegaconf import DictConfig
from typing_extensions import override

from icenet_mp.types import DataSpace, ModelStepOutput, TensorNTCHW

from .base_model import BaseModel

if TYPE_CHECKING:
    from icenet_mp.models.decoders import BaseDecoder
    from icenet_mp.models.encoders import BaseEncoder
    from icenet_mp.models.processors import BaseProcessor


class EncodeProcessDecode(BaseModel):
    """Model that encodes to latent space, processes, then decodes back."""

    ignored_hparams: ClassVar[frozenset[str]] = BaseModel.ignored_hparams | {"mask_dir"}

    def __init__(
        self,
        *,
        encoders: DictConfig,
        processor: DictConfig,
        decoder: DictConfig,
        target_variable_indices: list[int],
        mask_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise an EncodeProcessDecode model."""
        super().__init__(**kwargs)

        if self.output_space.channels != len(target_variable_indices):
            msg = (
                f"output_space has {self.output_space.channels} channel(s) but "
                f"target_variable_indices selects {len(target_variable_indices)}; "
                f"check that predict.target.variables is set correctly."
            )
            raise ValueError(msg)
        self.target_variable_indices = target_variable_indices

        try:
            self.encoders: list[BaseEncoder] = [
                hydra.utils.instantiate(
                    encoders[input_space.name],
                    data_space_in=input_space,
                    latent_space=encoders["latent_space"],
                    latitudes_fn=self.latitudes_fn,
                    longitudes_fn=self.longitudes_fn,
                )
                for input_space in self.input_spaces
            ]
        except KeyError as exc:
            msg = (
                f"Error instantiating encoders: {exc}. Please ensure that encoders are "
                f"specified for all input spaces: {self.input_spaces}"
            )
            raise ValueError(msg) from exc

        try:
            self.target_encoder: BaseEncoder = hydra.utils.instantiate(
                encoders[self.output_space.name],
                data_space_in=DataSpace(
                    name="target",
                    channels=self.output_space.channels,
                    shape=self.output_space.shape,
                ),
                latent_space=encoders["latent_space"],
                latitudes_fn=self.latitudes_fn,
                longitudes_fn=self.longitudes_fn,
            )
        except KeyError as exc:
            msg = (
                f"Error instantiating target encoder: {exc}. Please ensure that an "
                f"encoder is specified for '{self.output_space.name}', even if it is "
                f"not one of the input spaces: {self.input_spaces}."
            )
            raise ValueError(msg) from exc

        for input_space, module in zip(self.input_spaces, self.encoders, strict=True):
            module_name = f"encoder_{input_space.name}".lower().replace("-", "_")
            self.add_module(module_name, module)

        latent_shapes = {encoder.data_space_out.shape for encoder in self.encoders}
        if len(latent_shapes) != 1:
            msg = (
                f"Expected all encoders to have the same output shape, but found "
                f"{len(latent_shapes)} different shapes: {latent_shapes}"
            )
            raise ValueError(msg)

        for encoder in (*self.encoders, self.target_encoder):
            encoder.verify_output_channels(self.device)

        combined_latent_space = DataSpace(
            name="combined_latent_space",
            channels=sum(encoder.data_space_out.channels for encoder in self.encoders),
            shape=latent_shapes.pop(),
        )
        target_channel_offset = self.find_target_channel_offset()
        target_input_space = next(
            (
                input_space
                for input_space in self.input_spaces
                if input_space.name == self.output_space.name
            ),
            None,
        )
        target_group_channels = (
            target_input_space.channels if target_input_space is not None else None
        )

        self.processor: BaseProcessor = hydra.utils.instantiate(
            processor,
            data_space=combined_latent_space,
            data_space_target=self.target_encoder.data_space_out,
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_channel_offset=target_channel_offset,
        )

        self.decoder: BaseDecoder = hydra.utils.instantiate(
            decoder,
            data_space_in=combined_latent_space,
            data_space_out=self.output_space,
            mask_dir=mask_dir,
            target_channel_offset=target_channel_offset,
            target_group_channels=target_group_channels,
            target_variable_indices=self.target_variable_indices,
        )

        self._freeze_unused_modules()

    @property
    def multistage_only(self) -> bool:
        return self.processor.computes_loss_in_latent_space

    def _freeze_unused_modules(self) -> None:
        """Freeze unused modules."""
        if self.processor.computes_loss_in_latent_space:
            self.decoder.freeze()
        else:
            self.target_encoder.freeze()

    def encode_inputs(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Encode all input datasets and concatenate along the channel dimension."""
        latent_inputs: list[TensorNTCHW] = [
            encoder.rollout(inputs[encoder.name]) for encoder in self.encoders
        ]
        return torch.cat(latent_inputs, dim=2)

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Run the encode-process-decode inference path."""
        latent_input_combined: TensorNTCHW = self.encode_inputs(inputs)
        latent_output: TensorNTCHW = self.processor.rollout(
            latent_input_combined
        ).prediction
        persistence = self.get_persistence(inputs)
        return self.decoder.rollout(latent_output, persistence)

    def find_target_channel_offset(self) -> int | None:
        """Find the target dataset offset within concatenated latent channels."""
        offset = 0
        for encoder, input_space in zip(self.encoders, self.input_spaces, strict=True):
            if input_space.name == self.output_space.name:
                return offset
            offset += encoder.data_space_out.channels
        return None

    def get_persistence(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW | None:
        """Extract persistence if needed for a skip connection."""
        if self.decoder.skip_connection:
            return inputs[self.output_space.name][
                :, -1, self.target_variable_indices, :, :
            ].unsqueeze(1)
        return None

    @override
    def train(self, mode: bool = True) -> "EncodeProcessDecode":
        """Set training mode, preserving any processor-specific freezing."""
        super().train(mode)
        if mode:
            self._freeze_unused_modules()
        return self

    def training_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,
    ) -> ModelStepOutput:
        """Run a training step."""
        batch = self.process_batch(batch)
        target = batch["target"].clone().detach()
        combined_latent = self.encode_inputs(batch)

        expected_chw = self.target_encoder.data_space_in.chw
        if tuple(target.shape[2:]) != expected_chw:
            msg = (
                f"Target CHW {tuple(target.shape[2:])} does not match "
                f"'{self.target_encoder.name}' encoder input (C, H, W)={expected_chw}."
            )
            raise ValueError(msg)

        target_latent = None
        if self.processor.computes_loss_in_latent_space:
            target_latent = self.target_encoder.rollout(target)
        processor_output = self.processor.rollout(combined_latent, target_latent)

        persistence = self.get_persistence(batch)

        if processor_output.loss is None:
            prediction = self.decoder.rollout(processor_output.prediction, persistence)
            loss = self.loss(prediction, target)
        else:
            loss = processor_output.loss
            with torch.no_grad():
                prediction = self.decoder.rollout(
                    processor_output.prediction, persistence
                )

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