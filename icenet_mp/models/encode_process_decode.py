from typing import TYPE_CHECKING, Any, ClassVar, cast

import hydra
import torch
from omegaconf import DictConfig

from icenet_mp.types import DataSpace, TensorNCHW, TensorNTCHW

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
        zero_init_tendency: bool = True,
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
            **kwargs: forwarded to ``BaseModel`` (spaces, steps, optimiser, loss, ...).

        Args:
            rollout_space: "latent" (default, unchanged behaviour) or "physical" (residual learning, autoregressive-like).

                "latent" option is the original method: the processor predicts the next
                combined latent, which is then appended to its own input window
                (BaseProcessor.rollout). Potentiallly two flagged consequences, from tested
                failures (eg, on 2026-07-29): (1) no constrains on a processor to produce a
                latent to lie in the distribution the encoders produce, and there is no
                re-encoding (consistency term), so from forecast step 2 onward the
                window is fed vectors from a different distribution; (2) the roll-out window
                takes every input group's latent channels and carry them forward in time, so a
                multi-input model forecasts uncontraint ERA5/Argo latents and then
                conditions on them. The optimiser tends to find the easiest "optimum" to satisfying both, which is a near-idempotent map, which likely explains the almost fixed-point (static forecast) behaviour observed.

                "physical" option wraps the loop in observation space instead: encode the
                window of physical frames, take one processor step (roll-out and forecast),
                decode back to a physical field, roll that field into the window, and re-encode.
                The feedback state in this option is always something the encoders were trained
                on. Groups other than the SIC (target) keep their last observed frame (we
                dont have atmospheric forecast; persisting is an assumption and we can maybe improve this, but it should prevent a hallucinated latent that might get trapped on persistent latent for the reason given above).

            predict_residual: if True the decoder output is a TENDENCY added to the
                previous field, so `prediction = clamp(previous + delta, 0, 1)`.

                This is the fix for the binding failure: with the absolute
                parameterisation, reproducing the input requires pushing it through a
                downsampling CNN, a patch-embedding ViT bottleneck and an upsampling
                CNN, so persistence is not in the model's hypothesis space. Measured
                on the toy, the model scores active-cell MAE 0.093 at lead 1 where
                merely copying the input scores 0.034. With a residual head, delta = 0
                is exactly the persistence, and every parameter is trained to predict
                the change instead of rebuilding the persistence state. This Requires decoder's
                restrict_range to be set to "none": the bounding is applied to the sum here,
                not to the increment.

            zero_init_tendency: with predict_residual=True, zero the decoder final
                convolution at initialisation so the model starts at persistence exactly,
                and training can move away from it only to reduce the loss (intended).
                Otherwise a random initialisation of the tendency produces a large signed
                field, the sum saturates against the [0, 1] clamp, and the first epochs
                are spent climbing back down to sanity state (this is also observed in real training runs), a mechanism that might be partially responsible for many previous runs ended up worse than persistence.

            feedback_channel: index of the channel within the target input group (SIC) that
                the prediction overwrites when it is rolled back into the window under the
                physical rollout option. This is only needed when the target group has more channels
                than the model outputs (eg, production sic-ssmis has 6, the output has 1);
                when the counts match, all channels are replaced.

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

        # Add a decoder
        self.decoder: BaseDecoder = hydra.utils.instantiate(
            decoder,
            data_space_in=combined_latent_space,
            data_space_out=self.output_space,
            mask_dir=mask_dir,
        )

        # Start the trajectory at exactly the persistence: a zeroed tendency head means the first
        # forward pass reproduces the last observation exactly at every lead.
        if self.predict_residual and zero_init_tendency:
            self._zero_tendency_head()

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

        if str(decoder.get("restrict_range") or "none") != "none":
            msg = (
                f"predict_residual=True requires the decoder's restrict_range to be "
                f"'none' (got {decoder.get('restrict_range')!r}): the decoder emits a "
                f"signed tendency, which must not be squashed into [0, 1]. The sum "
                f"previous + tendency is clamped to [0, 1] by the model instead."
            )
            raise ValueError(msg)

    def _zero_tendency_head(self) -> None:
        """Zero the decoder's output convolution so the initial tendency is exactly 0."""
        final = cast("torch.nn.Sequential", self.decoder.model)[-1]
        if not isinstance(final, torch.nn.Conv2d):
            msg = (
                f"zero_init_tendency=True requires the decoder to end in a Conv2d so the "
                f"tendency can be 0, but {type(self.decoder).__name__} ends in "
                f"{type(final).__name__}. Try pass zero_init_tendency=false to skip this."
            )
            raise TypeError(msg)
        with torch.no_grad():
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model.

        - start with multiple [NTCHW] inputs each with shape [batch, n_history_steps, n_input_channels_k, H_input_k, W_input_k]
        - encode inputs to [NTCHW] latent space [batch, n_history_steps, n_latent_channels, H_latent, W_latent]
        - concatenate inputs in [NTCHW] latent space [batch, n_history_steps, n_latent_channels_total, H_latent, W_latent]
        - process in latent space [NTCHW] [batch, n_forecast_steps, n_latent_channels_total, H_latent, W_latent]
        - decode back to [NTCHW] output space [batch, n_forecast_steps, n_output_channels, H_output, W_output]
        - add a skip connection from the most recent target value to every forecast step

        When `rollout_space="physical"` the loop is closed in observation space instead;
        see `_forward_physical`.
        """
        if self.rollout_space == "physical":
            return self._forward_physical(inputs)

        # Encode inputs into latent space: list of tensors with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)
        latent_inputs: list[TensorNTCHW] = [
            encoder.rollout(inputs[encoder.name]) for encoder in self.encoders
        ]

        # Combine in the variable dimension: tensor with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
        latent_input_combined: TensorNTCHW = torch.cat(latent_inputs, dim=2)

        # Process in latent space:
        # combined input tensor with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
        latent_output: TensorNTCHW = self.processor.rollout(
            latent_input_combined
        ).prediction

        # Add persistence skip connection if requested
        persistence: TensorNTCHW | None = (
            inputs[self.output_space.name][
                :, -1, self.target_variable_indices, :, :
            ].unsqueeze(1)
            if self.decoder.skip_connection
            else None
        )

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
