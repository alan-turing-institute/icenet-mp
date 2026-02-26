import os
from typing import Any, NoReturn

import torch
import torch.nn.functional as F  # noqa: N812
from torchmetrics import Metric, MetricCollection

from icenet_mp.losses import WeightedMSELoss
from icenet_mp.metrics import IceNetAccuracy, SIEError
from icenet_mp.models.diffusion import GaussianDiffusion, UNetDiffusion
from icenet_mp.types import ModelTestOutput, TensorNTCHW

from .base_model import BaseModel

# Unset SLURM_NTASKS if it's causing issues
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]

# Optionally, set SLURM_NTASKS_PER_NODE if needed
os.environ["SLURM_NTASKS_PER_NODE"] = "1"

# Force all new tensors to be float32 by default
torch.set_default_dtype(torch.float32)


class SimpleEncoder2D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Simple 2D encoder block using Conv2d, GroupNorm, and SiLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        """
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.GroupNorm(4, out_channels),
            torch.nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor after applying the block.

        """
        return self.net(x)


class DDPM(BaseModel):
    """Denoising Diffusion Probabilistic Model (DDPM).

    Input space:
        TensorNTCHW with shape (batch_size, n_history_steps + n_history_steps * n_era5_channels, height, width)
        - OSISAF input: T historical steps, singleton channel squeezed
        - ERA5 input: T historical steps times number of channels, resized to OSISAF resolution

    Output space:
        TensorNTCHW with shape (batch_size, n_forecast_steps * n_output_channels, height, width)
        - Forecasted outputs per timestep and channel, flattened along the channel dimension
    """

    def __init__(  # noqa: PLR0913
        self,
        timesteps: int = 1000,
        learning_rate: float = 5e-4,
        start_out_channels: int = 32,
        kernel_size: int = 3,
        activation: str = "SiLU",
        normalization: str = "groupnorm",
        time_embed_dim: int = 256,
        dropout_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize the DDPM processor.

        Args:
            timesteps (int): Number of diffusion timesteps. Default is 1000.
            learning_rate (float): Optimizer learning rate for training. Default is 5e-4.
            start_out_channels (int): Base number of channels in the first UNet block.
            kernel_size (int): Convolution kernel size used in the UNet.
            activation (str): Activation function used throughout the network (e.g., "SiLU").
            normalization (str): Normalization layer type (e.g., "groupnorm").
            time_embed_dim (int): Dimensionality of the timestep embedding.
            dropout_rate (float): Dropout probability applied inside the UNet blocks.
            **kwargs: Additional arguments passed to ``BaseModel``.

        """
        super().__init__(**kwargs)

        self.osisaf_key = self.output_space.name

        era5_space = next(
            space
            for space in self.input_spaces
            if (space["name"] if isinstance(space, dict) else space.name) == "era5"
        )

        # Get channels from either dict or object
        if isinstance(era5_space, dict):
            self.era5_space = era5_space["channels"]
        else:
            self.era5_space = era5_space.channels

        # Get the base output channels from output_space
        if isinstance(self.output_space, dict):
            self.base_output_channels = self.output_space["channels"]
        else:
            self.base_output_channels = self.output_space.channels

        self.output_channels = self.n_forecast_steps * self.base_output_channels
        self.timesteps = timesteps
        self.cond_channels = 64
        self.input_channels = self.cond_channels

        # "InstanceNorm" calculates the mean/std per batch, removing the need for offline preprocessing
        self.era5_norm = torch.nn.InstanceNorm3d(self.era5_space, affine=True)

        # Reduces the many ERA5 channels down to 32 important ones using 1x1 Conv
        self.era5_compressed_channels = 32
        self.era5_projector = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.era5_space, self.era5_compressed_channels, kernel_size=1
            ),
            torch.nn.SiLU(),
        )

        self.osisaf_encoder = SimpleEncoder2D(
            in_channels=self.n_history_steps,
            out_channels=self.cond_channels // 2,
        )

        # (Compressed_Channels * Time_Steps), preserving time history
        self.era5_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.era5_compressed_channels * self.n_history_steps,
                out_channels=self.cond_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.GroupNorm(4, self.cond_channels // 2),
            torch.nn.SiLU(),
        )

        self.model = UNetDiffusion(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            timesteps=self.timesteps,
            kernel_size=kernel_size,
            start_out_channels=start_out_channels,
            time_embed_dim=time_embed_dim,
            normalization=normalization,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        self.diffusion = GaussianDiffusion(timesteps=timesteps)

        self.learning_rate = learning_rate

        metrics: dict[str, Metric | MetricCollection] = {
            "val_accuracy": IceNetAccuracy(
                leadtimes_to_evaluate=list(range(self.n_forecast_steps))
            ),
            "val_sieerror": SIEError(
                leadtimes_to_evaluate=list(range(self.n_forecast_steps))
            ),
        }
        for i in range(self.n_forecast_steps):
            metrics[f"val_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
            metrics[f"val_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        self.metrics = MetricCollection(metrics)

        test_metrics: dict[str, Metric | MetricCollection] = {
            "test_accuracy": IceNetAccuracy(
                leadtimes_to_evaluate=list(range(self.n_forecast_steps))
            ),
            "test_sieerror": SIEError(
                leadtimes_to_evaluate=list(range(self.n_forecast_steps))
            ),
        }
        for i in range(self.n_forecast_steps):
            test_metrics[f"test_accuracy_{i}"] = IceNetAccuracy(
                leadtimes_to_evaluate=[i]
            )
            test_metrics[f"test_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        self.test_metrics = MetricCollection(test_metrics)

        self.save_hyperparameters()

    def forward(self, *args: Any, **kwargs: Any) -> NoReturn:
        msg = "This model uses `training_step`, `validation_step`, and `test_step` instead of `forward()`"
        raise NotImplementedError(msg)

    def sample(
        self,
        x: torch.Tensor,
        sample_weight: torch.Tensor | None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Perform reverse diffusion sampling starting from noise.

        Args:
            x (torch.Tensor): Conditioning input [B, C, H, W].
            sample_weight (torch.Tensor or None): Optional weights.

        Returns:
            torch.Tensor: Denoised output of shape [B, C, H, W].

        """
        shape = (
            x.shape[0],
            self.n_forecast_steps * self.base_output_channels,
            *x.shape[-2:],
        )

        # Start from pure noise
        y = torch.randn(shape, device=self.device)

        dim_threshold = 3

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full_like(
                x[:, 0, 0, 0], t, dtype=torch.long, device=self.device
            )
            pred_v: torch.Tensor = self.model(y, t_batch, x)
            pred_v = (
                pred_v.squeeze(3) if pred_v.dim() > dim_threshold else pred_v.squeeze()
            )
            y = self.diffusion.p_sample(y, t_batch, pred_v)

        return y

    def loss(
        self,
        prediction: TensorNTCHW,
        target: TensorNTCHW,
        sample_weight: TensorNTCHW | None = None,
    ) -> torch.Tensor:
        if sample_weight is None:
            sample_weight = torch.ones_like(prediction)
        return WeightedMSELoss(reduction="none")(prediction, target, sample_weight)

    def prepare_inputs(self, batch: dict[str, TensorNTCHW]) -> torch.Tensor:
        """Encode OSISAF and ERA5 separately, then concatenate.

        ERA5 -> Norm -> Project -> Resize -> Flatten Time -> Encode

        Args:
            batch: Dictionary with
                'osisaf-south' [B, T, 1, H, W]
                'era5' [B, T, C, H2, W2]

        Returns:
            Conditioning tensor [B, cond_channels, H, W]

        """
        osisaf = batch[self.osisaf_key]  # [B, T, 1, H, W]
        era5 = batch["era5"]  # [B, T, C, H2, W2]

        # Handle OSISAF
        osisaf = osisaf.squeeze(2)  # [B, T, H, W]
        H, W = osisaf.shape[-2:]  # noqa: N806

        # Handle ERA5
        # Permute to [B, C, T, H2, W2] for 3D operations
        era5 = era5.permute(0, 2, 1, 3, 4)

        # Normalize (On-the-fly standardization)
        era5 = self.era5_norm(era5)

        # Project (Learnable Feature Selection)
        # [B, C, T, H2, W2] -> [B, 32, T, H2, W2]
        era5 = self.era5_projector(era5)

        # Resize Spatially
        B, C_new, T, H2, W2 = era5.shape  # noqa: N806
        # Flatten batch/channel/time for interpolation
        era5_flat = era5.reshape(B * C_new * T, 1, H2, W2)

        era5_resized = F.interpolate(
            era5_flat,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # Flatten Time into Channels
        # Reshape back to [B, C_new, T, H, W] then flatten T into C
        era5_features = era5_resized.reshape(B, C_new * T, H, W)

        # Encode Both
        osisaf_features = self.osisaf_encoder(osisaf)  # [B, cond//2, H, W]

        # era5_features enters the encoder as a 2D tensor with many channels
        era5_features = self.era5_encoder(era5_features)  # [B, cond//2, H, W]

        return torch.cat([osisaf_features, era5_features], dim=1)  # [B, cond, H, W]

    def training_step(
        self, batch: dict[str, TensorNTCHW], _batch_idx: int
    ) -> torch.Tensor:
        """One training step using DDPM v-prediction loss.

        During training, the clean target (SIC) is corrupted using the forward
        diffusion process by adding noise at a randomly sampled timestep.
        The model is trained to predict the corresponding v-target.

        Args:
            batch (dict[str, TensorNTCHW]):
                Dictionary containing:
                    - input tensors (used to prepare conditioning inputs)
                    - "target": groundtruth SIC tensor

        Returns:
            torch.Tensor:
                Scalar training loss (MSE between predicted v and target v).

        """
        # Prepare input tensor by combining osisaf-south and era5
        x = self.prepare_inputs(batch)  # [B, T, C_combined, H, W]

        # Extract target
        y = batch["target"].squeeze(2)

        # Sample random timesteps
        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=self.device
        ).long()  # look into this

        # Create noisy version
        noise = torch.randn_like(y)
        noisy_y = self.diffusion.q_sample(y, t, noise)

        # Predict v
        pred_v = self.model(noisy_y, t, x)

        # Compute target v
        target_v = self.diffusion.calculate_v(y, noise, t)

        # Compute loss
        loss = F.mse_loss(pred_v, target_v)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: dict[str, TensorNTCHW], _batch_idx: int
    ) -> torch.Tensor:
        """One validation step using full diffusion sampling.

        During validation, samples are generated by starting from noise and
        iteratively denoising conditioned on the inputs. The final prediction
        is compared to the groundtruth SIC using the configured evaluation loss.

        Args:
            batch (dict[str, TensorNTCHW]):
                Dictionary containing:
                    - input tensors (used to prepare conditioning inputs)
                    - "target": groundtruth SIC tensor
                    - optional "sample_weight": weighting tensor

        Returns:
            torch.Tensor:
                Scalar validation loss computed between predicted SIC (y_hat)
                and groundtruth SIC (y).

        """
        # Prepare input tensor
        x = self.prepare_inputs(batch)  # [B, T, C_combined, H, W]

        # Extract target and optional weights
        y = batch["target"].squeeze(2)  # [B, T, H, W]
        sample_weight = batch.get("sample_weight", torch.ones_like(y))

        # Generate samples
        outputs = self.sample(x, sample_weight)

        y_hat = torch.clamp(outputs, 0, 1)

        # Calculate loss
        loss = self.loss(y_hat, y, sample_weight)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Update metrics
        self.metrics.update(y_hat, y, sample_weight)
        return loss

    def test_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,  # noqa: PT019
    ) -> ModelTestOutput:
        """One test step using full diffusion sampling and metric evaluation.

        During testing, predictions are generated by starting from noise
        and running the reverse diffusion process conditioned on the inputs.
        The final reconstructed SIC is compared to the groundtruth target
        using the configured loss and test metrics.

        Args:
            batch (dict[str, TensorNTCHW]):
                Dictionary containing:
                    - input tensors (used to prepare conditioning inputs)
                    - "target": groundtruth SIC tensor
                    - optional "sample_weight": weighting tensor

        Returns:
            ModelTestOutput:
                Object containing:
                    - prediction: reconstructed SIC (y_hat)
                    - target: groundtruth SIC (y)
                    - loss: test loss value

        """
        x = self.prepare_inputs(batch)  # [B, T, C_combined, H, W]
        y = batch["target"].squeeze(2)
        sample_weight = batch.get("sample_weight", torch.ones_like(y))

        outputs = self.sample(x, sample_weight)

        y_hat = torch.clamp(outputs, 0, 1).unsqueeze(2)

        y = y.unsqueeze(2)
        sample_weight = sample_weight.unsqueeze(2)

        loss = self.loss(y_hat, y, sample_weight)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.test_metrics.update(y_hat, y, sample_weight)

        return ModelTestOutput(prediction=y_hat, target=y, loss=loss)
