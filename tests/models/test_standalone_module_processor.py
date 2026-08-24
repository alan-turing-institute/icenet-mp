import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from icenet_mp.models.processors import StandaloneModuleProcessor
from icenet_mp.types import DataSpace


class NonTensorModule(nn.Module):
    """Small invalid module used to exercise output validation."""

    def forward(self, _x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a deliberately invalid non-tensor output."""
        return {"prediction": torch.tensor(0.0)}


def _processor(module: nn.Module | DictConfig) -> StandaloneModuleProcessor:
    """Build an adapter around a two-history-step latent model."""
    return StandaloneModuleProcessor(
        module=module,
        data_space=DataSpace(name="latent", channels=2, shape=(8, 8)),
        n_forecast_steps=3,
        n_history_steps=2,
        target_channel_offset=0,
    )


def test_standalone_adapter_matches_wrapped_module_forward() -> None:
    """Return exactly the same one-step output as the standalone module."""
    module = nn.Conv2d(4, 2, kernel_size=1)
    processor = _processor(module)
    x = torch.randn(3, 4, 8, 8)

    expected = module(x)
    actual = processor(x)

    torch.testing.assert_close(actual, expected)


def test_standalone_adapter_rollout_and_gradients() -> None:
    """Use the normal autoregressive rollout and preserve module gradients."""
    module = nn.Conv2d(4, 2, kernel_size=1)
    processor = _processor(module)
    history = torch.randn(3, 2, 2, 8, 8, requires_grad=True)

    prediction = processor.rollout(history).prediction
    prediction.square().mean().backward()

    assert prediction.shape == (3, 3, 2, 8, 8)
    assert history.grad is not None
    assert module.weight.grad is not None
    assert torch.isfinite(module.weight.grad).all()


def test_standalone_adapter_instantiates_hydra_module_config() -> None:
    """Accept a nested Hydra configuration for an existing standalone module."""
    processor = _processor(
        DictConfig(
            {
                "_target_": "torch.nn.Conv2d",
                "in_channels": 4,
                "out_channels": 2,
                "kernel_size": 1,
            }
        )
    )

    assert isinstance(processor.module, nn.Conv2d)
    assert processor(torch.randn(1, 4, 8, 8)).shape == (1, 2, 8, 8)


def test_standalone_adapter_rejects_wrong_output_shape() -> None:
    """Reject standalone models that violate the processor latent shape contract."""
    processor = _processor(nn.Conv2d(4, 3, kernel_size=1))

    with pytest.raises(ValueError, match="expected"):
        processor(torch.randn(1, 4, 8, 8))


def test_standalone_adapter_rejects_non_tensor_output() -> None:
    """Reject standalone models that do not return a tensor."""
    processor = _processor(NonTensorModule())

    with pytest.raises(TypeError, match="must return a torch.Tensor"):
        processor(torch.randn(1, 4, 8, 8))
