import hydra
import pytest
import torch
from omegaconf import OmegaConf
from torch.nn import functional

from icenet_mp.losses.amse_loss import AMSELoss, AMSEMode


def make_fields(
    seed: int = 0, shape: tuple[int, ...] = (2, 2, 1, 32, 32)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a seeded (prediction, target) pair of random fields."""
    generator = torch.Generator().manual_seed(seed)
    prediction = torch.rand(*shape, generator=generator)
    target = torch.rand(*shape, generator=generator)
    return prediction, target


def blur(field: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Box-blur the trailing spatial dimensions of an [N, T, C, H, W] tensor."""
    n, t, c, h, w = field.shape
    flat = field.reshape(n * t * c, 1, h, w)
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size**2
    blurred = functional.conv2d(flat, kernel, padding=kernel_size // 2)
    return blurred.reshape(n, t, c, h, w)


class TestAMSELoss:
    def test_config_instantiation(self) -> None:
        config = OmegaConf.create(
            {
                "_target_": "icenet_mp.losses.amse_loss.AMSELoss",
                "mode": "hybrid",
                "spectral_weight": 0.1,
            }
        )
        loss_fn = hydra.utils.instantiate(config)
        assert isinstance(loss_fn, AMSELoss)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            AMSELoss(mode="not-a-mode")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="merge_bins_below"):
            AMSELoss(merge_bins_below=0)

    @pytest.mark.parametrize("mode", ["hybrid", "pure"])
    def test_identity_is_zero(self, mode: AMSEMode) -> None:
        prediction, _ = make_fields()
        loss_fn = AMSELoss(mode=mode)
        assert loss_fn(prediction, prediction).item() == pytest.approx(0.0, abs=1e-6)

    def test_scalar_output(self) -> None:
        prediction, target = make_fields()
        loss = AMSELoss()(prediction, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_parseval_identity(self) -> None:
        """Binned spectra must reconstruct the spatial MSE exactly (Parseval)."""
        prediction, target = make_fields(seed=1)
        loss_fn = AMSELoss()
        power_p, power_t, cross, dc = loss_fn.binned_spectra(prediction, target)
        spectral_mse = (power_p + power_t - 2.0 * cross).sum(dim=1) + dc
        spatial_mse = (
            ((prediction - target) ** 2).reshape(spectral_mse.shape[0], -1).mean(dim=1)
        )
        assert torch.allclose(spectral_mse, spatial_mse, rtol=1e-4, atol=1e-7)

    def test_parseval_identity_odd_grid(self) -> None:
        """The rfft factor-2 bookkeeping must also hold for odd widths."""
        prediction, target = make_fields(seed=2, shape=(2, 1, 1, 31, 33))
        loss_fn = AMSELoss()
        power_p, power_t, cross, dc = loss_fn.binned_spectra(prediction, target)
        spectral_mse = (power_p + power_t - 2.0 * cross).sum(dim=1) + dc
        spatial_mse = (
            ((prediction - target) ** 2).reshape(spectral_mse.shape[0], -1).mean(dim=1)
        )
        assert torch.allclose(spectral_mse, spatial_mse, rtol=1e-4, atol=1e-7)

    def test_blur_is_penalized_and_monotone(self) -> None:
        """The spectral excess is positive for blurred predictions and grows with blur width."""
        _, target = make_fields(seed=3)
        loss_fn = AMSELoss()
        excess_light = loss_fn.spectral_excess(blur(target, 3), target).mean()
        excess_heavy = loss_fn.spectral_excess(blur(target, 7), target).mean()
        assert excess_light.item() > 0.0
        assert excess_heavy.item() > excess_light.item()

    def test_amplitude_gradient_points_up(self) -> None:
        """For a uniformly damped prediction the spectral gradient pushes amplitude up."""
        _, target = make_fields(seed=4)
        scale = torch.tensor(0.5, requires_grad=True)
        excess = AMSELoss().spectral_excess(scale * target, target).mean()
        (gradient,) = torch.autograd.grad(excess, scale)
        assert gradient.item() < 0.0

    @pytest.mark.parametrize("mode", ["hybrid", "pure"])
    def test_zero_field_gradients_finite(self, mode: AMSEMode) -> None:
        _, target = make_fields(seed=5)
        prediction = torch.zeros_like(target, requires_grad=True)
        loss = AMSELoss(mode=mode)(prediction, target)
        loss.backward()
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()

    @pytest.mark.parametrize("mode", ["hybrid", "pure"])
    def test_masked_fields(self, mode: AMSEMode) -> None:
        """Fields zeroed outside an active region give finite loss and gradients."""
        prediction, target = make_fields(seed=6)
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1.0
        prediction = (prediction * mask).detach().requires_grad_()
        target = target * mask
        loss = AMSELoss(mode=mode)(prediction, target)
        loss.backward()
        assert torch.isfinite(loss)
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()

    def test_spectral_part_is_symmetric(self) -> None:
        prediction, target = make_fields(seed=7)
        loss_fn = AMSELoss(mode="pure")
        assert torch.allclose(loss_fn(prediction, target), loss_fn(target, prediction))

    def test_hybrid_reduces_to_huber_at_zero_weight(self) -> None:
        prediction, target = make_fields(seed=8)
        hybrid = AMSELoss(mode="hybrid", spectral_weight=0.0)(prediction, target)
        huber = functional.huber_loss(prediction, target, delta=0.5)
        assert torch.allclose(hybrid, huber)

    def test_low_precision_input(self) -> None:
        prediction, target = make_fields(seed=9)
        loss = AMSELoss()(prediction.to(torch.bfloat16), target.to(torch.bfloat16))
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)
