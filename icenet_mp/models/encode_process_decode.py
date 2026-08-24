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
        use_motion_channels: bool = False,
        use_day_order_channels: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialise an EncodeProcessDecode model.

        Args:
            encoders: Per-dataset encoder configs, plus `latent_space`.
            processor: Processor config.
            decoder: Decoder config.
            mask_dir: Optional directory containing land/active masks.
            use_motion_channels: If True, concatenate frame-to-frame differences of
                each input dataset's history window onto its raw frames (channel dim)
                before encoding, giving every encoder an explicit velocity cue instead
                of relying on it being inferred purely from stacked raw frames. The
                first history step has no earlier frame to diff against, so its motion
                channels are zero-filled. Doubles each encoder's expected input
                channel count.
            use_day_order_channels: If True, concatenate one extra channel onto each
                timestep of each input dataset's history window before encoding,
                holding a constant label for that timestep's position in the window
                (0 for the oldest day up to 1 for the most recent). This gives the
                encoder an explicit day-order cue rather than relying on order being
                inferred from channel position alone. Adds one to each encoder's
                expected input channel count.
            target_variable_indices: Channel indices of the target variables within
                the target group's input tensor.

        """
        super().__init__(**kwargs)
        self.use_motion_channels = use_motion_channels
        self.use_day_order_channels = use_day_order_channels

        def encoder_input_space(input_space: DataSpace) -> DataSpace:
            channels = input_space.channels
            if use_motion_channels:
                channels *= 2
            if use_day_order_channels:
                channels += 1
            if channels == input_space.channels:
                return input_space
            return DataSpace(
                channels=channels,
                name=input_space.name,
                shape=input_space.shape,
            )

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
                    data_space_in=encoder_input_space(input_space),
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
        self.processor: BaseProcessor = hydra.utils.instantiate(
            processor,
            data_space=combined_latent_space,
            data_space_target=self.target_encoder.data_space_out,
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

    @staticmethod
    def _with_motion_channels(x: TensorNTCHW) -> TensorNTCHW:
        """Concatenate frame-to-frame differences as extra channels (explicit velocity cue).

        The first history step has no earlier frame to diff against, so its motion
        channels are zero-filled rather than dropped, to keep the timestep count
        unchanged.
        """
        diffs = x[:, 1:] - x[:, :-1]
        zero_pad = torch.zeros_like(x[:, :1])
        diffs = torch.cat([zero_pad, diffs], dim=1)
        return torch.cat([x, diffs], dim=2)

    @staticmethod
    def _with_day_order_channels(x: TensorNTCHW) -> TensorNTCHW:
        """Concatenate a constant day-order label channel onto each timestep.

        Each timestep gains one channel whose value marks its position in the history
        window, spaced evenly from 0 (oldest) to 1 (most recent). A single-timestep
        window is labelled 0.
        """
        batch, n_steps, _, height, width = x.shape
        labels = torch.linspace(
            0.0, 1.0, max(n_steps, 2), device=x.device, dtype=x.dtype
        )[:n_steps]
        labels = labels.view(1, n_steps, 1, 1, 1).expand(
            batch, n_steps, 1, height, width
        )
        return torch.cat([x, labels], dim=2)

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
        latent_inputs: list[TensorNTCHW] = []
        for encoder in self.encoders:
            tensor = inputs[encoder.name]
            if self.use_motion_channels:
                tensor = self._with_motion_channels(tensor)
            if self.use_day_order_channels:
                tensor = self._with_day_order_channels(tensor)
            latent_inputs.append(encoder.rollout(tensor))
        return torch.cat(latent_inputs, dim=2)

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model (used for inference).

        - start with multiple [NTCHW] inputs each with shape [batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k]
        - encode inputs to [NTCHW] latent space [batch, n_history_steps, n_latent_channels, H_latent, W_latent]
        - concatenate inputs in [NTCHW] latent space [batch, n_history_steps, n_latent_channels_total, H_latent, W_latent]
        - process in latent space [NTCHW] [batch, n_forecast_steps, n_latent_channels_total, H_latent, W_latent]
        - decode back to [NTCHW] output space [batch, n_forecast_steps, n_output_channels, H_output, W_output]
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

    def find_target_channel_offset(self) -> int | None:
        """Find the channel offset of the target dataset within the combined latent space, if present."""
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
