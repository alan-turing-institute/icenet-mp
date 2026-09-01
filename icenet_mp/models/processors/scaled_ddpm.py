"""DDPM processor with configurable clean-target scaling."""

from typing import Any

from icenet_mp.types import ProcessorOutput, TensorNCHW, TensorNTCHW

from .ddpm import DDPMProcessor


class ScaledDDPMProcessor(DDPMProcessor):
    """DDPM processor that scales clean target latents before diffusion.

    ``x0_scale`` should be set to the standard deviation of the clean target
    latent used for the experiment. Dividing by this value makes the clean
    target approximately unit variance, while predictions are multiplied by
    the same value before they are returned to the decoder.

    A value of 1.0 is equivalent to the existing ``DDPMProcessor`` behaviour.
    """

    def __init__(self, *, x0_scale: float = 1.0, **kwargs: Any) -> None:
        """Initialise the scaled DDPM processor.

        Args:
            x0_scale: Positive scale applied to clean target latents. Use the
                target latent standard deviation to normalise x0 to unit
                variance. Defaults to 1.0, which leaves the latent unchanged.
            **kwargs: Arguments forwarded to ``DDPMProcessor``.

        Raises:
            ValueError: If ``x0_scale`` is not positive.

        """
        if x0_scale <= 0:
            msg = f"x0_scale must be positive, got {x0_scale}."
            raise ValueError(msg)
        self.x0_scale = x0_scale
        super().__init__(**kwargs)

    def _rollout_training(self, x: TensorNTCHW, y: TensorNTCHW) -> ProcessorOutput:
        """Run training with x0 scaled before the forward diffusion process."""
        output = super()._rollout_training(x, y / self.x0_scale)

        # Metrics and the decoder operate in the original latent scale.
        prediction = output.prediction.clone()
        prediction[..., self.target_slice_start : self.target_slice_end, :, :] *= (
            self.x0_scale
        )
        return ProcessorOutput(prediction=prediction, loss=output.loss)

    def _run_reverse_diffusion(self, y: TensorNCHW, cond: TensorNCHW) -> TensorNCHW:
        """Denoise in unit-scale space and return the original latent scale."""
        prediction = super()._run_reverse_diffusion(y, cond)
        return prediction * self.x0_scale
