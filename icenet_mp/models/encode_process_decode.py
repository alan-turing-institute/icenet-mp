from typing import TYPE_CHECKING, Any

import hydra
import torch
from omegaconf import DictConfig

from icenet_mp.types import DataSpace, TensorNTCHW

from .base_model import BaseModel

if TYPE_CHECKING:
    from icenet_mp.models.decoders import BaseDecoder
    from icenet_mp.models.encoders import BaseEncoder
    from icenet_mp.models.processors import BaseProcessor


class EncodeProcessDecode(BaseModel):
    def __init__(
        self,
        *,
        encoders: DictConfig,
        processor: DictConfig,
        decoder: DictConfig,
        mask_dir: str | None = None,
        use_skip_connection: bool = False,
        use_motion_channels: bool = False,
        use_day_order_channels: bool = False,
        use_residual_output: bool = False,
        target_variable_indices: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise an EncodeProcessDecode model.

        Args:
            encoders: Per-dataset encoder configs, plus `latent_space`.
            processor: Processor config.
            decoder: Decoder config.
            mask_dir: Optional directory containing land/active masks.
            use_skip_connection: If True, concatenate the most recent encoded history
                timestep onto the processor's output before decoding (channel dim),
                broadcasting it across all forecast steps. Processors that route
                everything through a bottleneck with no spatial skip path of their own
                (e.g. a patchified ViT) lose fine boundary detail that a skip
                connection lets the decoder recover directly from the input, the same
                way UNetProcessor's own internal skip connections do.
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
            use_residual_output: If True, the decoded output is treated as a *change*
                relative to the most recent observed target frame (persistence) rather
                than the full field: final output = clamp(last_frame + delta, 0, 1).
                Routing only the delta through the processor bottleneck means a
                zero-output network already matches persistence, so capacity is spent
                on what changes instead of reconstructing the whole field through the
                patch bottleneck. The target's group must be one of the input datasets.
                Pair this with ``decoder.restrict_range: none`` -- the delta must be
                allowed to go negative; the [0, 1] clamp here bounds the final sum.
            target_variable_indices: Channel indices of the target variables within
                the target group's input tensor, used to select the persistence base
                frame when ``use_residual_output`` is set (ModelService passes this
                automatically). Defaults to the first ``output_space.channels``
                channels if not given.

        """
        super().__init__(**kwargs)
        self.use_skip_connection = use_skip_connection
        self.use_motion_channels = use_motion_channels
        self.use_day_order_channels = use_day_order_channels
        self.use_residual_output = use_residual_output
        if use_residual_output:
            input_names = {space.name for space in self.input_spaces}
            if self.output_space.name not in input_names:
                msg = (
                    f"use_residual_output requires the target group "
                    f"'{self.output_space.name}' to be one of the input datasets "
                    f"{sorted(input_names)}, so the last observed frame is available "
                    f"as the persistence base."
                )
                raise ValueError(msg)
            indices = (
                list(target_variable_indices)
                if target_variable_indices is not None
                else list(range(self.output_space.channels))
            )
            if len(indices) != self.output_space.channels:
                msg = (
                    f"target_variable_indices selects {len(indices)} channels but the "
                    f"output space has {self.output_space.channels}."
                )
                raise ValueError(msg)
            self.residual_variable_indices = indices

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

        # Add a processor
        combined_latent_space = DataSpace(
            name="combined_latent_space",
            channels=sum(encoder.data_space_out.channels for encoder in self.encoders),
            shape=latent_shapes.pop(),
        )
        self.processor: BaseProcessor = hydra.utils.instantiate(
            processor,
            data_space=combined_latent_space,
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
        )

        # Add a decoder. If a skip connection is used, the decoder receives the
        # processor's output concatenated with the most recent encoded history
        # timestep, so it has twice as many input channels.
        decoder_data_space_in = (
            DataSpace(
                name=combined_latent_space.name,
                channels=combined_latent_space.channels * 2,
                shape=combined_latent_space.shape,
            )
            if use_skip_connection
            else combined_latent_space
        )
        self.decoder: BaseDecoder = hydra.utils.instantiate(
            decoder,
            data_space_in=decoder_data_space_in,
            data_space_out=self.output_space,
            mask_dir=mask_dir,
        )

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

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        - start with multiple [NTCHW] inputs each with shape [batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k]
        - encode inputs to [NTCHW] latent space [batch, n_history_steps, n_latent_channels, H_latent, W_latent]
        - concatenate inputs in [NTCHW] latent space [batch, n_history_steps, n_latent_channels_total, H_latent, W_latent]
        - process in latent space [NTCHW] [batch, n_forecast_steps, n_latent_channels_total, H_latent, W_latent]
        - decode back to [NTCHW] output space [batch, n_forecast_steps, n_output_channels, H_output, W_output]
        """
        encoder_inputs = inputs
        if self.use_motion_channels:
            encoder_inputs = {
                name: self._with_motion_channels(tensor)
                for name, tensor in encoder_inputs.items()
            }
        if self.use_day_order_channels:
            encoder_inputs = {
                name: self._with_day_order_channels(tensor)
                for name, tensor in encoder_inputs.items()
            }

        # Encode inputs into latent space: list of tensors with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)
        latent_inputs: list[TensorNTCHW] = [
            encoder.rollout(encoder_inputs[encoder.name]) for encoder in self.encoders
        ]

        # Combine in the variable dimension: tensor with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
        latent_input_combined: TensorNTCHW = torch.cat(latent_inputs, dim=2)

        # Process in latent space:
        # combined input tensor with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
        latent_output: TensorNTCHW = self.processor.rollout(
            latent_input_combined
        ).prediction

        # Optionally skip the most recent encoded history timestep straight to the
        # decoder (bypassing the processor), broadcast across every forecast step.
        if self.use_skip_connection:
            most_recent = latent_input_combined[:, -1:, :, :, :]
            skip = most_recent.expand(-1, self.n_forecast_steps, -1, -1, -1)
            latent_output = torch.cat([latent_output, skip], dim=2)

        # Decode to output space: tensor with (batch_size, n_forecast_steps, n_output_channels, output_height, output_width)
        output: TensorNTCHW = self.decoder.rollout(latent_output)

        # In residual mode the decoder output is a delta from persistence: add the
        # most recent *raw* observed target frame (not the encoded one) and bound the
        # sum to the valid concentration range.
        if self.use_residual_output:
            base = inputs[self.output_space.name][
                :, -1:, self.residual_variable_indices, :, :
            ]
            output = (output + base).clamp(min=0.0, max=1.0)

        # Return
        return output
