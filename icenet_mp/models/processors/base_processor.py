from torch import nn, stack

from icenet_mp.types import DataSpace, ModelStepOutput, TensorNCHW, TensorNTCHW


class BaseProcessor(nn.Module):
    """Processor that converts latent input into latent output.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        data_space: DataSpace,
        n_forecast_steps: int,
        n_history_steps: int,
    ) -> None:
        """Initialise a BaseProcessor."""
        super().__init__()
        self.data_space = data_space
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: process in NCHW latent space for a single timestep.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        """
        msg = "If you are using the default forward method, you must implement rollout."
        raise NotImplementedError(msg)

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ModelStepOutput:
        """Process in latent space across multiple timesteps.

        The default implementation simply calls `self.forward` on each time slice until
        a sufficient number of forecast steps have been produced. These are then stacked
        together to produce the final output.

        Override this method to handle the NTCHW tensors directly or to compute a custom
        loss using the target tensor `y` (e.g. for diffusion models).

        Args:
            x: Encoded input TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
            y: during training: Encoded target TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
                otherwise: None

        Returns:
            ModelStepOutput with:
              prediction: TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
              target: the target tensor (if provided)
              loss: custom loss computed by the processor (if implemented)

        """
        # Cut the NTCHW input into NCHW slices
        nchw_slices = [x[:, idx_t, :, :, :] for idx_t in range(self.n_history_steps)]

        # Rollout the model over the input slices, producing an output for each one.
        # Also append the predictions to the list of input slices, so that we can still
        # predict when n_forecast_steps > n_history_steps.
        outputs: list[TensorNCHW] = []
        for _ in range(self.n_forecast_steps):
            outputs.append(self(nchw_slices.pop(0)))
            nchw_slices.append(outputs[-1])

        return ModelStepOutput(prediction=stack(outputs, dim=1), target=y, loss=None)
