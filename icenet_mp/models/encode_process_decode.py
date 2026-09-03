from typing import TYPE_CHECKING, Any, ClassVar

import hydra
import torch
from omegaconf import DictConfig
from typing_extensions import override

from icenet_mp.types import DataSpace, ModelStepOutput, TensorNCHW, TensorNTCHW

from .base_model import BaseModel

if TYPE_CHECKING:
    from icenet_mp.models.decoders import BaseDecoder
    from icenet_mp.models.encoders import BaseEncoder
    from icenet_mp.models.processors import BaseProcessor


class EncodeProcessDecode(BaseModel):
    """Model that encodes to latent space, processes, then decodes back."""

    # Parameters that should be excluded from hyperparameter logging (e.g. local paths)
    ignored_hparams: ClassVar[frozenset[str]] = BaseModel.ignored_hparams | {"mask_dir"}

    def __init__(  # noqa: PLR0913 - config-driven keywords, all defaulted
        self,
        *,
        encoders: DictConfig,
        processor: DictConfig,
        decoder: DictConfig,
        target_variable_indices: list[int],
        mask_dir: str | None = None,
        rollout_space: str = "latent",
        predict_residual: bool = False,
        feedback_channel: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise an EncodeProcessDecode model.

        Args:
            encoders: config for the per-input-group encoders (plus ``latent_space``).
            processor: config for the latent-space processor.
            decoder: config for the decoder producing the output space.
            target_variable_indices: indices, within the target INPUT group, of the
                variable(s) being predicted. Must select exactly as many channels as
                ``output_space`` has; used to pull the newest observed target frame
                out of the input window as the skip connection's anchor.
            mask_dir: directory holding the mask ``.npy`` files, if masking is used.
            rollout_space: which space the autoregressive forecast loop closes in.

                ``"latent"`` (default): the processor rolls forward in latent space,
                appending each predicted latent to its own input window without
                re-encoding (``BaseProcessor.rollout``). All input groups' latent
                channels are carried forward, so non-target groups are implicitly
                forecast in latent space as well.

                ``"physical"``: each forecast step encodes the current window of
                physical frames, takes one processor step, decodes to a physical
                field and rolls that field back into the window, which is re-encoded
                on the next step (``_forward_physical``). The fed-back state is
                always a physical field. Non-target groups (e.g. ERA5/Argo) hold
                their newest observed frame for every step of the rollout, i.e.
                their latents are replaced with persistence at each step.

            predict_residual: if True the decoder output is a signed TENDENCY and the
                prediction is ``previous + delta``, applied by the decoder's additive
                skip connection, so a zero tendency reproduces the previous field
                exactly. Requires ``rollout_space="physical"`` (the residual is added
                to the previous physical field) and a decoder configured with
                ``skip_connection.method: additive``. The anchor is the previous
                forecast step's field, updated every step — unlike the latent path's
                skip connection, which anchors every lead on the newest observation.
            feedback_channel: index of the channel within the target input group that
                the prediction overwrites when it is rolled back into the window
                under the physical rollout. Only needed when the target group has
                more channels than the model outputs (e.g. a 6-channel sic-ssmis
                input with a 1-channel output); when the counts match, all channels
                are replaced.
            **kwargs: forwarded to ``BaseModel`` (spaces, steps, optimiser, loss, ...).

        """
        super().__init__(**kwargs)

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

        self._validate_rollout_options(rollout_space, predict_residual, decoder)
        self.rollout_space = rollout_space
        self.predict_residual = bool(predict_residual)
        self.feedback_channel = feedback_channel

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

        # The physical rollout drives the decoder itself and computes the loss on the
        # decoded field, so it cannot host a processor that owns the training signal
        # in latent space (whose decoder is frozen below).
        if rollout_space == "physical" and self.processor.computes_loss_in_latent_space:
            msg = (
                f"rollout_space='physical' is incompatible with processor "
                f"{type(self.processor).__name__}, which computes its loss in latent "
                f"space (the decoder is frozen and never trained on that path)."
            )
            raise ValueError(msg)

        # Freeze unused modules
        self._freeze_unused_modules()

    @staticmethod
    def _validate_rollout_options(
        rollout_space: str,
        predict_residual: bool,  # noqa: FBT001 - mirrors the __init__ keyword
        decoder: DictConfig,
    ) -> None:
        """Reject rollout/residual settings that cannot work, before anything is built.

        Kept out of `__init__` so that adding a check does not push it past the
        cyclomatic-complexity limit.
        """
        if rollout_space not in {"latent", "physical"}:
            msg = (
                f"rollout_space must be 'latent' or 'physical', got {rollout_space!r}."
            )
            raise ValueError(msg)
        if not predict_residual:
            return

        if rollout_space != "physical":
            msg = (
                "predict_residual=True requires rollout_space='physical': the residual "
                "is added to the previous PHYSICAL field, which only exists as a "
                "rollout state in physical space."
            )
            raise ValueError(msg)

        # The residual update is applied by the decoder's additive skip connection
        # (#405), so one must be configured: finalise() adds the anchor only when
        # skip_connection is present, and would otherwise silently drop it and
        # return an absolute prediction (rather than the tendency).
        skip_method = str((decoder.get("skip_connection") or {}).get("method", "none"))
        if skip_method != "additive":
            msg = (
                f"predict_residual=True requires the decoder to use an additive skip "
                f"connection (got skip_connection.method={skip_method!r}): the "
                f"tendency is added to the anchor by the decoder's skip connection, "
                f"so without it the anchor would be dropped and the output would be "
                f"an absolute prediction rather than a residual one."
            )
            raise ValueError(msg)

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

        - start with multiple [NTCHW] inputs each with shape [batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k]
        - encode inputs to [NTCHW] latent space [batch, n_history_steps, n_latent_channels, H_latent, W_latent]
        - concatenate inputs in [NTCHW] latent space [batch, n_history_steps, n_latent_channels_total, H_latent, W_latent]
        - process in latent space [NTCHW] [batch, n_forecast_steps, n_latent_channels_total, H_latent, W_latent]
        - decode back to [NTCHW] output space [batch, n_forecast_steps, n_output_channels, H_output, W_output]
        - add a skip connection from the most recent target value to every forecast step

        When `rollout_space="physical"` the loop is closed in observation space instead;
        see `_forward_physical`.
        """
        # getattr because multistage's ProcessorStage reuses this method without
        # running EncodeProcessDecode.__init__ (same reason as in training_step).
        if getattr(self, "rollout_space", "latent") == "physical":
            return self._forward_physical(inputs)

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

    def _forward_physical(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Autoregressive-like rollout closed in physical/observation space one forecast step at a time.

        Per step: encode the window of observed/predicted frames, take ONE processor
        step, decode to a physical field, then roll that field into the window and
        re-encode on the next iteration. Contrast with the default path, which appends
        the processor's raw latent to its own input window and never re-encodes.

        With `predict_residual=True` the decoder emits a tendency and the state advances
        as `x_{k+1} = clamp(x_k + delta_k, 0, 1)`, so a zero-output network reproduces
        persistence exactly.

        Non-target input groups hold their most recent OBSERVED frame for every forecast
        step. No future information enters: only `inputs[...]`, which holds the
        n_history_steps frames ending at the forecast origin, is ever read; the ground
        truth lives under the separate reserved key "target" and is not touched here.
        """
        target_name = self.output_space.name
        if target_name not in inputs:
            msg = (
                f"rollout_space='physical' requires the prediction target group "
                f"'{target_name}' to also be a model input, so that the rollout has an "
                f"observed physical state to advance. Inputs: {sorted(inputs)}."
            )
            raise ValueError(msg)

        n_out = self.output_space.channels
        target_window = inputs[target_name].clone()  # (B, nh, C_t, H, W)
        n_target_channels = target_window.shape[2]
        if self.feedback_channel is None and n_target_channels != n_out:
            msg = (
                f"The target group '{target_name}' has {n_target_channels} channels but "
                f"the model outputs {n_out}; set model.feedback_channel to the index of "
                f"the channel the prediction should overwrite when it is fed back."
            )
            raise ValueError(msg)

        # Non-target groups: hold the newest observed frame for the whole rollout.
        frozen: dict[str, TensorNTCHW] = {
            name: tensor[:, -1:].expand_as(tensor)
            for name, tensor in inputs.items()
            if name not in {target_name, "target"}
        }

        outputs: list[TensorNCHW] = []
        for _ in range(self.n_forecast_steps):
            windows = dict(frozen)
            windows[target_name] = target_window

            latent = torch.cat(
                [encoder.rollout(windows[encoder.name]) for encoder in self.encoders],
                dim=2,
            )  # (B, nh, C_latent_total, h, w)

            # One processor step: the window is concatenated along channels, oldest to
            # newest, exactly as BaseProcessor.rollout does it.
            step_in = torch.cat(
                [latent[:, idx_t] for idx_t in range(self.n_history_steps)], dim=1
            )
            step_latent = self.processor(step_in)

            raw = self.decoder(step_latent)

            if self.predict_residual:
                # Compared to the latent path the difference here is
                # the anchor: here we use the state produced by the previous forecast step.
                anchor = self._anchor(target_window, n_out)
                field = self.decoder.finalise(raw, anchor)
            else:
                # The non-residual physical path has no
                # anchor of its own, so pass None; the decoder then applies only
                # range restriction and masking, exactly as before.
                field = self.decoder.finalise(raw, None)
            outputs.append(field)

            target_window = self._advance(target_window, field)

        return torch.stack(outputs, dim=1)

    def _anchor(self, target_window: TensorNTCHW, n_out: int) -> TensorNCHW:
        """Return the current physical state that a residual is added to."""
        newest = target_window[:, -1]  # (B, C_t, H, W)
        if self.feedback_channel is None:
            return newest
        idx = int(self.feedback_channel)
        return newest[:, idx : idx + n_out]

    def _advance(self, target_window: TensorNTCHW, field: TensorNCHW) -> TensorNTCHW:
        """Drop the oldest target frame and append the newly predicted one."""
        newest = target_window[:, -1].clone()
        if self.feedback_channel is None:
            newest = field
        else:
            idx = int(self.feedback_channel)
            newest[:, idx : idx + field.shape[1]] = field
        return torch.cat([target_window[:, 1:], newest.unsqueeze(1)], dim=1)

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

        # The physical rollout lives in forward(). Without this branch the model would
        # TRAIN on the latent path below while validation_step/test_step (BaseModel,
        # which call ``self(batch)``) score the PHYSICAL path - two different
        # architectures, silently. Pinned by TestTrainEvalParity. getattr because
        # multistage's ProcessorStage reuses this method without EPD's __init__.
        if getattr(self, "rollout_space", "latent") == "physical":
            prediction = self(batch)
            loss = self.loss(prediction, target)
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
