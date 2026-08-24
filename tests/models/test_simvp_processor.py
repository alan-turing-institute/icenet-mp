from importlib.resources import files

import pytest
import torch
import yaml

from icenet_mp.models.processors import SimVPProcessor
from icenet_mp.types import DataSpace


def _processor(**kwargs: int) -> SimVPProcessor:
    """Build a small SimVP processor for unit tests."""
    config = {"hidden_channels": 8, "n_mixer_blocks": 2}
    config.update(kwargs)
    return SimVPProcessor(
        data_space=DataSpace(name="latent", channels=3, shape=(8, 8)),
        n_forecast_steps=2,
        n_history_steps=3,
        **config,
    )


def test_simvp_rollout_preserves_processor_shape() -> None:
    """Return one latent frame for every requested forecast step."""
    processor = _processor()
    inputs = torch.randn(2, 3, 3, 8, 8)

    output = processor.rollout(inputs)

    assert output.prediction.shape == (2, 2, 3, 8, 8)


def test_simvp_backpropagates_through_spatial_and_temporal_blocks() -> None:
    """Keep the spatial encoder and temporal mixer trainable end to end."""
    processor = _processor()
    inputs = torch.randn(2, 3, 3, 8, 8, requires_grad=True)

    processor.rollout(inputs).prediction.square().mean().backward()

    assert inputs.grad is not None
    assert processor.spatial_encoder[0].weight.grad is not None
    mixer_gradients = [
        parameter.grad
        for parameter in processor.temporal_mixer.parameters()
        if parameter.grad is not None
    ]
    assert mixer_gradients
    assert any(torch.count_nonzero(gradient) > 0 for gradient in mixer_gradients)


def test_simvp_prediction_depends_on_history_order() -> None:
    """Treat the history axis as ordered temporal information."""
    torch.manual_seed(0)
    processor = _processor()
    inputs = torch.randn(1, 3, 3, 8, 8)

    forward = processor(inputs.reshape(1, 9, 8, 8))
    reversed_history = inputs.flip(1).reshape(1, 9, 8, 8)
    backward = processor(reversed_history)

    assert not torch.allclose(forward, backward)


def test_simvp_rejects_invalid_history_channel_count() -> None:
    """Reject tensors that do not contain the configured history window."""
    processor = _processor()

    with pytest.raises(ValueError, match="Expected 9 concatenated history channels"):
        processor(torch.randn(1, 6, 8, 8))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_channels": 0}, "hidden_channels"),
        ({"kernel_size": 2}, "positive odd"),
        ({"n_mixer_blocks": 0}, "n_mixer_blocks"),
    ],
)
def test_simvp_rejects_invalid_configuration(
    kwargs: dict[str, int], message: str
) -> None:
    """Validate processor hyperparameters at construction time."""
    with pytest.raises(ValueError, match=message):
        _processor(**kwargs)


def test_simvp_model_config_selects_processor() -> None:
    """Expose a ready-to-run CNN/SimVP/CNN model configuration."""
    config_path = files("icenet_mp.config") / "model" / "cnn_simvp_cnn.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["name"] == "cnn-simvp-cnn"
    assert config["processor"]["_target_"] == (
        "icenet_mp.models.processors.SimVPProcessor"
    )
