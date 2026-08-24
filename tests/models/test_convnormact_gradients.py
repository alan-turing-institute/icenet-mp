import pytest
import torch
from torch import nn

from icenet_mp.models.common import CommonConvBlock


@pytest.mark.parametrize("norm_type", ["batchnorm", "groupnorm", "none"])
@pytest.mark.parametrize("n_subblocks", [2, 3])
def test_stacked_convnormact_preserves_finite_nonzero_gradients(
    norm_type: str,
    n_subblocks: int,
) -> None:
    """Current ConvNormAct stack depths should not catastrophically lose gradients."""
    torch.manual_seed(0)
    block = CommonConvBlock(
        in_channels=8,
        out_channels=8,
        kernel_size=3,
        n_subblocks=n_subblocks,
        norm_type=norm_type,
        dropout_rate=0.0,
    )
    block.train()

    inputs = torch.randn(4, 8, 16, 16, requires_grad=True)
    target = torch.randn_like(inputs)
    loss = torch.nn.functional.mse_loss(block(inputs), target)
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.norm() > 0

    gradient_norms = []
    for module in block.modules():
        if isinstance(module, nn.Conv2d):
            assert module.weight.grad is not None
            assert torch.isfinite(module.weight.grad).all()
            norm = module.weight.grad.norm()
            assert norm > 0
            gradient_norms.append(norm)

    assert len(gradient_norms) == n_subblocks
    gradient_norms_tensor = torch.stack(gradient_norms)
    relative_minimum = gradient_norms_tensor.min() / gradient_norms_tensor.max()
    assert relative_minimum > 1e-6
