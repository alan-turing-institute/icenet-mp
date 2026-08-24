import pytest
import torch

from icenet_mp.models.encoders import PretrainedEmbeddingEncoder
from icenet_mp.types import DataSpace


def _space(channels: int = 4, shape: tuple[int, int] = (8, 8)) -> DataSpace:
    return DataSpace(name="embedding", channels=channels, shape=shape)


def test_pretrained_embedding_encoder_is_exact_passthrough_when_aligned() -> None:
    """Preserve an already aligned pretrained embedding exactly."""
    encoder = PretrainedEmbeddingEncoder(
        data_space_in=_space(),
        latent_space=(8, 8),
    )
    x = torch.randn(2, 4, 8, 8)

    output = encoder(x)

    torch.testing.assert_close(output, x)


def test_pretrained_embedding_encoder_resizes_to_latent_shape() -> None:
    """Resize embeddings to the requested shared latent geometry."""
    encoder = PretrainedEmbeddingEncoder(
        data_space_in=_space(shape=(6, 10)),
        latent_space=(12, 20),
    )
    x = torch.randn(2, 4, 6, 10)

    output = encoder(x)

    assert output.shape == (2, 4, 12, 20)


def test_pretrained_embedding_encoder_projects_channels_and_backpropagates() -> None:
    """Project embedding channels with a trainable 1x1 adaptor when requested."""
    encoder = PretrainedEmbeddingEncoder(
        data_space_in=_space(channels=5),
        latent_space=(8, 8),
        output_channels=3,
    )
    x = torch.randn(2, 5, 8, 8, requires_grad=True)

    output = encoder(x)
    output.square().mean().backward()

    assert output.shape == (2, 3, 8, 8)
    assert x.grad is not None
    projection = encoder.projection
    assert isinstance(projection, torch.nn.Conv2d)
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()


def test_pretrained_embedding_encoder_rollout_preserves_time_dimension() -> None:
    """Apply the embedding adaptor independently across all history timesteps."""
    encoder = PretrainedEmbeddingEncoder(
        data_space_in=_space(channels=2, shape=(4, 4)),
        latent_space=(8, 8),
        output_channels=3,
    )
    x = torch.randn(2, 5, 2, 4, 4)

    output = encoder.rollout(x)

    assert output.shape == (2, 5, 3, 8, 8)


def test_pretrained_embedding_encoder_rejects_unknown_interpolation_mode() -> None:
    """Reject unsupported spatial interpolation modes during construction."""
    with pytest.raises(ValueError, match="Unsupported interpolation_mode"):
        PretrainedEmbeddingEncoder(
            data_space_in=_space(),
            latent_space=(8, 8),
            interpolation_mode="area",  # type: ignore[arg-type]
        )
