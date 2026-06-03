from torch import Tensor, nn, stack

from icenet_mp.types import DataSpace, TensorNCHW, TensorNTCHW


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

    def custom_loss(self, x: TensorNTCHW, y: TensorNTCHW) -> Tensor | None:  # noqa: ARG002
        """Compute a custom training loss in latent space.

        Processors, like diffusion models, that compute loss internally should override
        this method.

        Args:
            x: Combined latent history TensorNTCHW with (batch_size, n_history_steps,
               n_latent_channels_total, latent_height, latent_width)
            y: Encoded target TensorNTCHW with (batch_size, n_forecast_steps,
               n_latent_channels_total, latent_height, latent_width)

        Returns:
            A loss Tensor, or None to use loss computed on the decoded output.

        """
        return None

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: process in NCHW latent space for a single timestep.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        """
        msg = "If you are using the default forward method, you must implement rollout."
        raise NotImplementedError(msg)

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> TensorNTCHW:  # noqa: ARG002
        """Process in latent space across multiple timesteps.

        The default implementation simply calls `self.forward` on each time slice until
        a sufficient number of forecast steps have been produced. These are then stacked
        together to produce the final output.

        If you want to handle the NTCHW tensors directly, or to use the target tensor for
        model training simply override this method in your child class.

        Args:
            x: Input TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
            y: during training: Target TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
               otherwise:       None

        Returns:
            Predicted TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)

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

        # Stack the outputs up as a new time dimension
        return stack(outputs, dim=1)
