"""Latent-space DDIM processor.

This module provides a Denoising Diffusion Implicit Models (DDIM) processor
on top of the DDPM training pipeline. Training is unchanged from
:class:`DDPMProcessor`. Only reverse-time sampling is replaced.

Instead of stepping through every one of the ``timesteps`` used in training,
DDIM sampling only visits ``ddim_steps`` of them, evenly spaced. Using fewer
steps means faster sampling, both when generating predictions and during
in-training validation, but can cost a bit of sample quality.

``eta`` controls how much randomness is added back in at each step. At
``eta=0`` sampling is fully deterministic, the same starting noise always
gives the same output. At ``eta=1`` it adds noise equivalent to standard
DDPM sampling. Values in between blend the two.
"""

from typing import Any

import torch

from icenet_mp.types import TensorNCHW

from .ddpm import DDPMProcessor


class DDIMProcessor(DDPMProcessor):
    """Latent-space DDIM processor with v-prediction.

    Reuses :class:`DDPMProcessor` for training (identical v-prediction loss)
    and overrides reverse-time sampling with the DDIM update, allowing a
    smaller number of sampling steps than were used during training.

    Input/output space matches :class:`DDPMProcessor`.
    """

    def __init__(
        self,
        *,
        ddim_steps: int = 50,
        eta: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the DDIM processor.

        Args:
            ddim_steps (int): Number of DDIM sampling steps. Must satisfy
                ``1 <= ddim_steps <= timesteps``. Default is 50.
            eta (float): Stochasticity coefficient. ``eta=0`` gives
                deterministic DDIM sampling; ``eta=1`` matches DDPM's per-step noise.
                Default is 0.0.
            **kwargs: Forwarded to :class:`DDPMProcessor` (``timesteps``,
                ``beta_schedule``, ``loss``, and all other DDPM/BaseProcessor
                arguments).

        """
        super().__init__(**kwargs)

        if not 1 <= ddim_steps <= self.timesteps:
            msg = (
                f"ddim_steps={ddim_steps} must be in the range "
                f"[1, timesteps={self.timesteps}]."
            )
            raise ValueError(msg)
        if eta < 0.0:
            msg = f"eta={eta} must be non-negative."
            raise ValueError(msg)

        self.ddim_steps = ddim_steps
        self.eta = eta

        # Evenly-spaced subset of trained timesteps, highest-first.
        # e.g. timesteps=1000, ddim_steps=50 -> [999, ..., 0].
        self._ddim_timesteps = torch.linspace(
            self.timesteps - 1, 0, ddim_steps, dtype=torch.long
        )

    def _run_reverse_diffusion(self, y: TensorNCHW, cond: TensorNCHW) -> TensorNCHW:
        """Iteratively denoise ``y`` using the DDIM sampler.

        Args:
            y (TensorNCHW): Noisy latent to denoise, of shape (B, C, H, W).
            cond (TensorNCHW): History condition folded to channels, of shape
                (B, T_hist * C_combined, H, W).

        Returns:
            TensorNCHW: Denoised latent of shape (B, C, H, W).

        """
        b = y.shape[0]
        device = y.device
        alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        ts = self._ddim_timesteps.to(device)

        for i in range(self.ddim_steps):
            t = ts[i]
            pred_v = self.model(y, t.expand(b), cond)

            alpha_bar_t = alphas_cumprod[t]
            sqrt_ab = alpha_bar_t.sqrt()
            sqrt_1mab = (1.0 - alpha_bar_t).sqrt()

            # Recover x_0 and epsilon from v-prediction.
            pred_x0 = sqrt_ab * y - sqrt_1mab * pred_v
            pred_eps = sqrt_1mab * y + sqrt_ab * pred_v

            # Final step targets x_0 (alpha_bar_prev = 1).
            alpha_bar_prev = (
                alpha_bar_t.new_ones(())
                if i == self.ddim_steps - 1
                else alphas_cumprod[ts[i + 1]]
            )

            sigma = (
                self.eta
                * ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)).sqrt()
                * (1.0 - alpha_bar_t / alpha_bar_prev).sqrt()
            )
            dir_coeff = (1.0 - alpha_bar_prev - sigma**2).clamp(min=0.0).sqrt()
            noise = torch.randn_like(y) if self.eta > 0.0 else 0.0

            y = alpha_bar_prev.sqrt() * pred_x0 + dir_coeff * pred_eps + sigma * noise

        return y
