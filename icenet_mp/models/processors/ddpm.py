from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch import nn


from icenet_mp.models.diffusion import GaussianDiffusion, UNetDiffusion
from icenet_mp.types import BetaSchedule, ProcessorOutput, TensorNCHW, TensorNTCHW

from .base_processor import BaseProcessor


class DDPMProcessor(BaseProcessor):
    """Latent-space DDPM processor with v-prediction.

    Input space:
        TensorNTCHW with shape (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
        - Concatenation of every encoder's latent output along the channel dimension
        - The target-latent slice is assumed to occupy channels [target_slice_start : target_slice_start + n_latent_channels_target]

    Output space:
        TensorNTCHW with shape (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
        - Target-latent slice contains the denoised prediction from reverse diffusion
        - Non-target channels are carried forward from the last observed history frame (persistence in latent)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        kernel_size: int = 3,
        start_out_channels: int = 64,
        time_embed_dim: int = 256,
        dropout_rate: float = 0.1,
        normalization: str = "groupnorm",
        activation: str = "SiLU",
        use_autoregressive: bool = True,
        target_slice_start: int = 0,
        loss: DictConfig | nn.Module,
        **kwargs: Any,
    ) -> None:
        """Initialize the DDPM processor.

        Args:
            timesteps (int): Number of diffusion timesteps. Default is 1000.
            beta_schedule (str): Beta schedule used by ``GaussianDiffusion``:
                ``"cosine"`` or ``"linear"``. Default is ``"cosine"``.
            kernel_size (int): Convolution kernel size used in the conditional UNet.
            start_out_channels (int): Base number of channels in the first UNet block.
            time_embed_dim (int): Dimensionality of the diffusion timestep embedding.
            dropout_rate (float): Dropout probability applied inside the UNet blocks.
            normalization (str): Normalization layer type (e.g., "groupnorm").
            activation (str): Activation function used throughout the network (e.g., "SiLU").
            use_autoregressive (bool): Whether to use autoregressive sampling and
                one-step training. Default is True.
            target_slice_start (int): Channel offset at which the target-latent
                slice appears inside the combined latent. Default 0 assumes the
                target encoder is listed first in the model config.
            loss (DictConfig | nn.Module): Loss module applied to (pred_v,
                target_v). Set this to ``${loss}`` in the yaml to reuse the
                top-level configured loss.
            **kwargs: Additional arguments passed to ``BaseProcessor``.

        """

        super().__init__(**kwargs)

        self.loss_fn: nn.Module = (
            loss if isinstance(loss, nn.Module) else hydra.utils.instantiate(loss)
        )
        
        c_combined = self.data_space.channels

        c_target = self.data_space_target.channels

        if target_slice_start < 0 or target_slice_start + c_target > c_combined:
            msg = (
                f"target_slice_start={target_slice_start} with target channels="
                f"{c_target} does not fit inside combined latent channels="
                f"{c_combined}."
            )
            raise ValueError(msg)

        self.timesteps = timesteps
        self.use_autoregressive = use_autoregressive
        
        self.target_slice_start = target_slice_start
        self.c_target = c_target
        self.c_combined = c_combined

        cond_channels = c_combined * self.n_history_steps

        diffused_channels = (
            c_target if use_autoregressive else c_target * self.n_forecast_steps
        )
        self.diffused_channels = diffused_channels

        self.model = UNetDiffusion(
            input_channels=cond_channels,
            output_channels=diffused_channels,
            timesteps=timesteps,
            kernel_size=kernel_size,
            start_out_channels=start_out_channels,
            time_embed_dim=time_embed_dim,
            normalization=normalization,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.diffusion = GaussianDiffusion(
            timesteps=timesteps,
            beta_schedule=BetaSchedule(beta_schedule),
        )

    def rollout(
        self,
        x: TensorNTCHW,
        y: TensorNTCHW | None = None,
        ) -> ProcessorOutput:
        if y is not None:
            return self._training_rollout(x, y)
        return self._inference_rollout(x)       

    def _build_metrics_prediction(
        self, pred_x0: TensorNCHW, last_frame: TensorNCHW
        ) -> TensorNTCHW:
        """Lift the training-time x_0 estimate into an NTCHW combined-latent tensor.

        The returned tensor is only used for metrics/callbacks (the actual
        training signal is the v-prediction MSE loss). Non-target channels are
        filled with the last observed frame (persistence in latent space).

        Args:
            pred_x0 (TensorNCHW): Predicted x_0 of shape (B, C_target, H, W)
                in autoregressive mode, or (B, n_forecast_steps * C_target, H, W)
                in parallel mode.
            last_frame (TensorNCHW): Last observed history frame of shape
                (B, C_combined, H, W).

        Returns:
            TensorNTCHW: Combined-latent tensor of shape
                (B, n_forecast_steps, C_combined, H, W).

        """
        b = pred_x0.shape[0]
        h, w = last_frame.shape[-2:]

        if self.use_autoregressive:
            per_step = self._insert_target(last_frame, pred_x0)
            return per_step.unsqueeze(1).expand(
                b, self.n_forecast_steps, self.c_combined, h, w
            )

        pred_x0_ntchw = pred_x0.reshape(
            b, self.n_forecast_steps, self.c_target, h, w
        )

        expanded_last = last_frame.unsqueeze(1).expand(
            b, self.n_forecast_steps, self.c_combined, h, w
        ).clone()
        s = self.target_slice_start
        expanded_last[:, :, s : s + self.c_target] = pred_x0_ntchw
        return expanded_last

    def _flatten_history(self, x: TensorNTCHW) -> TensorNCHW:
        """Fold the history time dimension into channels.

        Args:
            x (TensorNTCHW): History tensor of shape (B, T_hist, C, H, W).

        Returns:
            TensorNCHW: Tensor of shape (B, T_hist * C, H, W).

        """
        b, t, c, h, w = x.shape
        return x.reshape(b, t * c, h, w)

    def _slice_target(self, x: TensorNCHW) -> TensorNCHW:
        """Extract the target-latent channels from a combined-latent NCHW tensor.

        Args:
            x (TensorNCHW): Combined-latent tensor of shape (B, C_combined, H, W).

        Returns:
            TensorNCHW: Target-latent slice of shape (B, C_target, H, W).

        """
        s = self.target_slice_start
        return x[:, s : s + self.c_target]
        
    def _insert_target(
        self, base_frame: TensorNCHW, target: TensorNCHW
    ) -> TensorNCHW:
        """Return a copy of ``base_frame`` with its target slice overwritten by ``target``.

        Used to produce a C_combined-channel prediction for the frozen decoder
        while only forecasting the target-latent slice.

        Args:
            base_frame (TensorNCHW): Combined-latent frame of shape (B, C_combined, H, W).
            target (TensorNCHW): Target-latent slice of shape (B, C_target, H, W).

        Returns:
            TensorNCHW: Combined-latent frame of shape (B, C_combined, H, W).

        """
        result = base_frame.clone()
        s = self.target_slice_start
        result[:, s : s + self.c_target] = target
        return result
    
    def _training_rollout(
        self, x: TensorNTCHW, y: TensorNTCHW
        ) -> ProcessorOutput:
        b = x.shape[0]
        device = x.device
        
        last_frame = x[:, -1]  # (B, C_combined, H, W)

        cond = self._flatten_history(x)  # (B, T_hist * C_combined, H, W)

        if self.use_autoregressive:
            # AR trains on t=0 only.
            y_flat = y[:, 0]  # (B, C_target, H, W)
        else:
            y_flat = y.reshape(b, self.n_forecast_steps * self.c_target, *y.shape[-2:])

        t = torch.randint(0, self.timesteps, (b,), device=device).long()

        noise = torch.randn_like(y_flat)
        noisy_y = self.diffusion.q_sample(y_flat, t, noise)

        pred_v = self.model(noisy_y, t, cond)

        target_v = self.diffusion.calculate_v(y_flat, noise, t)

        loss = self.loss_fn(pred_v, target_v)

        with torch.no_grad():
            pred_x0 = self.diffusion.calculate_v(
                x_start=pred_v, noise=noisy_y, t=t
            )
            prediction = self._build_metrics_prediction(pred_x0, last_frame)

        return ProcessorOutput(prediction=prediction, loss=loss)

    def _inference_rollout(self, x: TensorNTCHW) -> ProcessorOutput:
        if self.use_autoregressive:
            return ProcessorOutput(prediction=self._sample_autoregressive(x))
        return ProcessorOutput(prediction=self._sample_parallel(x))

    def _sample_parallel(self, x: TensorNTCHW) -> TensorNTCHW:
         """Non-autoregressive (parallel) reverse diffusion sampling.

        This method generates the entire forecast sequence in a single
        diffusion process applied to one joint output tensor. The forecast
        steps are encoded as channels in a single tensor of shape
        (B, n_forecast_steps * C_target, H, W), then reshaped back to NTCHW
        and lifted into the combined latent for the frozen decoder.

        Args:
            x (TensorNTCHW): Encoded history tensor of shape
                (B, n_history_steps, n_latent_channels_total, H, W).

        Returns:
            TensorNTCHW: Denoised forecast latent of shape
                (B, n_forecast_steps, n_latent_channels_total, H, W).

        Notes:
            - Sampling starts from Gaussian noise.
            - The model predicts v-parameterization at each diffusion step.
            - All forecast steps are denoised together as a single object.

        """
        cond = self._flatten_history(x)

        b, _, h, w = cond.shape
        device = cond.device

        y = torch.randn(
            (b, self.n_forecast_steps * self.c_target, h, w), device=device
        )

        for t_step in reversed(range(self.timesteps)):
            t = torch.full((b,), t_step, dtype=torch.long, device=device)
            pred_v = self.model(y, t, cond)
            y = self.diffusion.p_sample(y, t, pred_v)

        y_ntchw = y.reshape(b, self.n_forecast_steps, self.c_target, h, w)
        last_frame = x[:, -1]  # (B, C_combined, H, W)
        combined = last_frame.unsqueeze(1).expand(
            b, self.n_forecast_steps, self.c_combined, h, w
        ).clone()
        
        s = self.target_slice_start
        combined[:, :, s : s + self.c_target] = y_ntchw
        return combined

    def _sample_autoregressive(self, x: TensorNTCHW) -> TensorNTCHW:
        history = x.clone()  # (B, T_hist, C_combined, H, W)
        b, _, _, h, w = x.shape
        device = x.device
        all_predictions: list[TensorNCHW] = []

        for _ in range(self.n_forecast_steps):
            cond_window = history[:, -self.n_history_steps :]
            cond = self._flatten_history(cond_window)  # (B, T_hist * C_combined, H, W)

            y = torch.randn((b, self.c_target, h, w), device=device)

            for t_step in reversed(range(self.timesteps)):
                t = torch.full((b,), t_step, dtype=torch.long, device=device)
                pred_v = self.model(y, t, cond)
                y = self.diffusion.p_sample(y, t, pred_v)

            all_predictions.append(y)

            last_frame = history[:, -1]  # (B, C_combined, H, W)
            new_frame = self._insert_target(last_frame, y).unsqueeze(1)
            history = torch.cat([history, new_frame], dim=1)

        target_ntchw = torch.stack(all_predictions, dim=1)  # (B, T_fcst, C_target, H, W)
        last_frame = x[:, -1]
        combined = last_frame.unsqueeze(1).expand(
            b, self.n_forecast_steps, self.c_combined, h, w
        ).clone()

        s = self.target_slice_start
        combined[:, :, s : s + self.c_target] = target_ntchw
        return combined
    