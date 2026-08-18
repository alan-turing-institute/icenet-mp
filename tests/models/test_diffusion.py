from pathlib import Path

import numpy as np
import pytest
import torch

from icenet_mp.models.common import Mask, RestrictRange
from icenet_mp.models.diffusion import GaussianDiffusion, UNetDiffusion
from icenet_mp.types import RangeRestriction


class TestGaussianDiffusion:
    def test_q_sample_t0_returns_clean_input(self) -> None:
        diffusion = GaussianDiffusion(timesteps=4)
        clean = torch.randn(2, 1, 4, 4)
        noise = torch.randn_like(clean)
        timesteps = torch.zeros(2, dtype=torch.long)

        sampled = diffusion.q_sample(clean, timesteps, noise)

        assert torch.equal(sampled, clean)

    def test_q_sample_matches_forward_equation(self) -> None:
        diffusion = GaussianDiffusion(timesteps=4)
        clean = torch.randn(2, 1, 4, 4)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([1, 3])
        sqrt_alpha = diffusion.sqrt_alphas_cumprod[timesteps].view(2, 1, 1, 1)
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alphas_cumprod[
            timesteps
        ].view(2, 1, 1, 1)

        sampled = diffusion.q_sample(clean, timesteps, noise)

        expected = sqrt_alpha * clean + sqrt_one_minus_alpha * noise
        assert torch.allclose(sampled, expected)

    def test_calculate_v_matches_definition(self) -> None:
        diffusion = GaussianDiffusion(timesteps=4)
        clean = torch.randn(2, 1, 4, 4)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([1, 3])
        sqrt_alpha = diffusion.sqrt_alphas_cumprod[timesteps].view(2, 1, 1, 1)
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alphas_cumprod[
            timesteps
        ].view(2, 1, 1, 1)

        velocity = diffusion.calculate_v(clean, noise, timesteps)

        expected = sqrt_alpha * noise - sqrt_one_minus_alpha * clean
        assert torch.allclose(velocity, expected)

    def test_p_sample_t0_uses_posterior_mean_without_noise(self) -> None:
        diffusion = GaussianDiffusion(timesteps=4)
        noisy = torch.randn(2, 1, 4, 4)
        predicted_v = torch.randn_like(noisy)
        timesteps = torch.zeros(2, dtype=torch.long)
        sqrt_alpha = diffusion.sqrt_alphas_cumprod[0]
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alphas_cumprod[0]
        predicted_clean = sqrt_alpha * noisy - sqrt_one_minus_alpha * predicted_v
        expected = (
            diffusion.posterior_mean_coef1[0] * predicted_clean
            + diffusion.posterior_mean_coef2[0] * noisy
        )

        sampled = diffusion.p_sample(noisy, timesteps, predicted_v)

        assert torch.allclose(sampled, expected)


class TestUNetDiffusion:
    def test_forward_preserves_requested_output_shape(self) -> None:
        model = UNetDiffusion(
            input_channels=4,
            output_channels=2,
            timesteps=4,
            start_out_channels=4,
            dropout_rate=0.0,
        )
        noise = torch.randn(2, 2, 16, 16)
        conditioning = torch.randn(2, 4, 16, 16)
        timesteps = torch.tensor([0, 3])

        output = model(noise, timesteps, conditioning)

        assert output.shape == noise.shape
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize(
        ("parameter", "value", "message"),
        [
            ("kernel_size", 0, "Kernel size must be greater than 0"),
            ("start_out_channels", 0, "Start out channels must be greater than 0"),
        ],
    )
    def test_invalid_construction_arguments_raise(
        self, *, parameter: str, value: int, message: str
    ) -> None:
        kwargs = {parameter: value}

        with pytest.raises(ValueError, match=message):
            UNetDiffusion(input_channels=4, output_channels=1, **kwargs)


class TestMask:
    def test_none_returns_input_unchanged(self) -> None:
        mask = Mask(mask_type=None, output_shape=(2, 2))
        values = torch.randn(1, 1, 2, 2)

        assert torch.equal(mask(values), values)

    def test_loaded_mask_is_applied(self, tmp_path: Path) -> None:
        np.save(
            tmp_path / "land_mask.npy",
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        mask = Mask(mask_type="land", output_shape=(2, 2), mask_dir=tmp_path)
        values = torch.ones(1, 1, 2, 2)

        result = mask(values)

        expected = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        assert torch.equal(result, expected)


class TestRestrictRange:
    def test_clamp_bounds_values(self) -> None:
        restrict = RestrictRange(RangeRestriction.CLAMP, min_val=0.0, max_val=1.0)
        values = torch.tensor([[[[-1.0, 0.25], [0.75, 2.0]]]])

        result = restrict(values)

        expected = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])
        assert torch.equal(result, expected)

    @pytest.mark.parametrize(
        "method", [RangeRestriction.SIGMOID, RangeRestriction.TANH]
    )
    def test_smooth_restrictions_stay_in_range(
        self, *, method: RangeRestriction
    ) -> None:
        restrict = RestrictRange(method, min_val=-2.0, max_val=3.0)
        values = torch.tensor([[[[-100.0, 0.0, 100.0]]]])

        result = restrict(values)

        assert torch.all(result >= -2.0)
        assert torch.all(result <= 3.0)
