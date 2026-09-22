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

    # Parameters that should be excluded from hyperparameter logging (e.g. local paths)
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
        super().__init__(mask_dir=mask_dir, **kwargs)

        # Check that the number of variable indices provided matches the number of
        # channels in the output space.
        if self.output_space.channels != len(target_variable_indices):
            msg = (
                f"output_space has {self.output_space.channels} channel(s) but "
                f"target_variable_indices selects {len(target_variable_indices)}; "
                f"check that predict.target.variables is set correctly."
            )
            raise ValueError(msg)
        self.target_variable_indices = target_variable_indices

        # Add one encoder per dataset
        # We store this as a list to ensure consistent ordering
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

        # Add an additional encoder that encodes the target dataset into latent space
        # This will be used by any processors that need to compute latent space losses.
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

        # We have to explicitly register each encoder as list[Module] will not be
        # automatically picked up by PyTorch
        for input_space, module in zip(self.input_spaces, self.encoders, strict=True):
            module_name = f"encoder_{input_space.name}".lower().replace("-", "_")
            self.add_module(module_name, module)

        # Confirm that all encoders have the same output shape
        latent_shapes = {encoder.data_space_out.shape for encoder in self.encoders}
        if len(latent_shapes) != 1:
            msg = (
                f"Expected all encoders to have the same output shape, but found "
                f"{len(latent_shapes)} different shapes: {latent_shapes}"
            )
            raise ValueError(msg)

        # Verify the output channels for each encoder
        for encoder in (*self.encoders, self.target_encoder):
            encoder.verify_output_channels(self.device)

        # Add a processor
        combined_latent_space = DataSpace(
            name="combined_latent_space",
            channels=sum(encoder.data_space_out.channels for encoder in self.encoders),
            shape=latent_shapes.pop(),
        )
        target_input_encoder = self.target_input_encoder
        target_latent_space = (
            target_input_encoder.data_space_out
            if target_input_encoder is not None
            else self.target_encoder.data_space_out
        )
        self.processor: BaseProcessor = hydra.utils.instantiate(
            processor,
            data_space=combined_latent_space,
            data_space_target=target_latent_space,
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_channel_offset=self.find_target_channel_offset(),
        )

        # Add a decoder
        self.decoder: BaseDecoder = hydra.utils.instantiate(
            decoder,
            data_space_in=combined_latent_space,
            data_space_out=self.output_space,
            mask_dir=mask_dir,
        )

        # Freeze unused modules
        self._freeze_unused_modules()

    @property
    def multistage_only(self) -> bool:
        return self.processor.computes_loss_in_latent_space

    def _freeze_unused_modules(self) -> None:
        """Freeze unused modules."""
        # Processors that compute loss in latent space do not touch the decoder.
        # However, processors that do not do this, do not touch the target_encoder.
        # We therefore explicitly freeze the unused modules.
        if self.processor.computes_loss_in_latent_space:
            self.decoder.freeze()
        else:
            self.target_encoder.freeze()

    def encode_inputs(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Encode all input datasets and concatenate along the channel dimension.

        Args:
            inputs: Dictionary with one TensorNTCHW entry per input dataset with shape (batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k)

        Returns:
            TensorNTCHW with shape (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

        """
        latent_inputs: list[TensorNTCHW] = [
            encoder.rollout(inputs[encoder.name]) for encoder in self.encoders
        ]
        return torch.cat(latent_inputs, dim=2)

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model (used for inference).

        - start with multiple `NTCHW` inputs each with shape (batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k)
        - encode inputs to `NTCHW` latent space (batch, n_history_steps, n_latent_channels, H_latent, W_latent)
        - concatenate inputs in `NTCHW` latent space (batch, n_history_steps, n_latent_channels_total, H_latent, W_latent)
        - process in latent space `NTCHW` (batch, n_forecast_steps, n_latent_channels_total, H_latent, W_latent)
        - decode back to `NTCHW` output space (batch, n_forecast_steps, n_output_channels, H_output, W_output)
        - add a skip connection from the most recent target value to every forecast step
        """
        # Encode inputs into latent space: tensor with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
        latent_input_combined: TensorNTCHW = self.encode_inputs(inputs)

        # Process in latent space: tensor with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
        latent_output: TensorNTCHW = self.processor.rollout(
            latent_input_combined
        ).prediction

        # Get persistence if required for skip connection
        persistence = self.get_persistence(inputs)

        # Decode to output space: tensor with (batch_size, n_forecast_steps, n_output_channels, output_height, output_width)
        return self.decoder.rollout(latent_output, persistence)

    @property
    def target_input_encoder(self) -> "BaseEncoder | None":
        """Return the input encoder whose dataset contains the forecast target."""
        for encoder, input_space in zip(self.encoders, self.input_spaces, strict=True):
            if input_space.name == self.output_space.name:
                return encoder
        return None

    def find_target_channel_offset(self) -> int | None:
        """Find the channel offset of the target dataset within the combined latent space, if present."""
        offset = 0
        for encoder, input_space in zip(self.encoders, self.input_spaces, strict=True):
            if input_space.name == self.output_space.name:
                return offset
            offset += encoder.data_space_out.channels
        return None

    def encode_target_latent(
        self, inputs: dict[str, TensorNTCHW], target: TensorNTCHW
    ) -> TensorNTCHW:
        """Encode a forecast target in the decoder-compatible latent space.

        When the target dataset is also an input, use that exact input encoder. The
        forecast tensor may contain only a subset of the target dataset's physical
        variables, so omitted variables are persisted from the last observed frame
        before encoding. This keeps supervision in the same latent coordinate system
        that the processor history and decoder use.

        If the target dataset is not an input, retain the standalone target encoder
        path used by processors that do not insert a target slice into the combined
        latent representation.
        """
        if tuple(target.shape[2:]) != self.output_space.chw:
            msg = (
                f"Target CHW {tuple(target.shape[2:])} does not match output space "
                f"(C, H, W)={self.output_space.chw}."
            )
            raise ValueError(msg)

        target_input_encoder = self.target_input_encoder
        if target_input_encoder is None:
            return self.target_encoder.rollout(target)

        target_input = inputs[self.output_space.name]
        if tuple(target_input.shape[2:]) != target_input_encoder.data_space_in.chw:
            msg = (
                f"Target input CHW {tuple(target_input.shape[2:])} does not match "
                f"'{target_input_encoder.name}' encoder input (C, H, W)="
                f"{target_input_encoder.data_space_in.chw}."
            )
            raise ValueError(msg)

        full_target = (
            target_input[:, -1:].expand(-1, target.shape[1], -1, -1, -1).clone()
        )
        full_target[:, :, self.target_variable_indices, :, :] = target
        return target_input_encoder.rollout(full_target)

    def get_persistence(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW | None:
        """Extract persistence if needed for a skip connection."""
        if self.decoder.skip_connection:
            return inputs[self.output_space.name][
                :, -1, self.target_variable_indices, :, :
            ].unsqueeze(1)
        return None

    @override
    def train(self, mode: bool = True) -> "EncodeProcessDecode":
        """Set training mode, with decoder frozen if computing loss in latent space."""
        super().train(mode)
        if mode:
            self._freeze_unused_modules()
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
        batch = self.process_batch(batch)
        target = batch["target"].clone().detach()
        combined_latent = self.encode_inputs(batch)

        if tuple(target.shape[2:]) != self.output_space.chw:
            msg = (
                f"Target CHW {tuple(target.shape[2:])} does not match output space "
                f"(C, H, W)={self.output_space.chw}."
            )
            raise ValueError(msg)

        target_latent = None
        if self.processor.computes_loss_in_latent_space:
            target_latent = self.encode_target_latent(batch, target)
        processor_output = self.processor.rollout(combined_latent, target_latent)

        # Get persistence if required for skip connection
        persistence = self.get_persistence(batch)

        if processor_output.loss is None:
            # Standard path: compare decoded output to target.
            prediction = self.decoder.rollout(processor_output.prediction, persistence)
            loss = self.loss(prediction, target)
        else:
            # Custom loss path: processor owns the training signal.
            # Decode under no_grad for metrics/callbacks only.
            loss = processor_output.loss
            with torch.no_grad():
                prediction = self.decoder.rollout(
                    processor_output.prediction, persistence
                )

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
