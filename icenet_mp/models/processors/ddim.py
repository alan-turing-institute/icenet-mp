from typing import Any

import torch

from icenet_mp.types import TensorNCHW

from .ddpm import DDPMProcessor


class DDIMProcessor(DDPMProcessor):
    def __init__(
        self,
        *,
        ddim_steps: int = 50,
        eta: float = 0.0,
        **kwargs: Any,
    ) -> None:
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