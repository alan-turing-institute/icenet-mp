from importlib.resources import files

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig


def _compose_spacetime_vit_config() -> DictConfig:
    """Compose the space-time ViT model through the normal Hydra config path."""
    config_dir = str(files("icenet_mp.config"))
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            return compose(
                config_name="sample",
                overrides=["model=cnn_spacetime_vit_cnn"],
            )
    finally:
        GlobalHydra.instance().clear()


def test_spacetime_vit_model_config_enables_missing_argo_strategy() -> None:
    """Check the model config wires both time and missing-data components."""
    config = _compose_spacetime_vit_config()

    assert config.model.name == "cnn-spacetime-vit-cnn"
    assert (
        config.model.processor._target_
        == "icenet_mp.models.processors.SpaceTimeVitProcessor"
    )
    assert (
        config.model.encoders["float-argo"]._target_
        == "icenet_mp.models.encoders.MissingDataCNNEncoder"
    )
    assert config.model.encoders[
        "float-argo"
    ].conditioning_dropout_probability == pytest.approx(0.15)
    assert config.model.encoders["float-argo"].missing_fill_value == pytest.approx(-1.0)


def test_spacetime_vit_config_runs_full_model_with_missing_argo() -> None:
    """Run configured encoders, processor, decoder and training loss end to end."""
    config = _compose_spacetime_vit_config()
    config.model.encoders.latent_space = [4, 4]
    config.model.encoders["float-argo"].conditioning_dropout_probability = 0.0
    config.model.processor.emb_dim = 8
    config.model.processor.heads = 2
    config.model.processor.mlp_dim = 16
    config.model.processor.patch_size = 2
    config.model.processor.spatial_depth = 1
    config.model.processor.temporal_depth = 1
    config.model.processor.forecast_spatial_depth = 1
    config.model.decoder.mask_type = None

    model = instantiate(
        config.model,
        hemisphere="north",
        input_spaces=[
            {"name": "float-argo", "channels": 2, "shape": [8, 8]},
            {"name": "sic-osisaf", "channels": 1, "shape": [8, 8]},
            {"name": "era5", "channels": 3, "shape": [8, 8]},
        ],
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
        lr_scheduler=DictConfig({}),
        n_forecast_steps=2,
        n_history_steps=3,
        optimizer=DictConfig({}),
        output_space={"name": "sic-osisaf", "channels": 1, "shape": [8, 8]},
        scheduler=DictConfig({}),
        target_variable_indices=[0],
    )

    batch_size = 2
    inputs = {
        "float-argo": torch.randn(batch_size, 3, 2, 8, 8),
        "sic-osisaf": torch.rand(batch_size, 3, 1, 8, 8),
        "era5": torch.randn(batch_size, 3, 3, 8, 8),
    }
    inputs["float-argo"][:, 0] = -1.0

    model.eval()
    with torch.no_grad():
        prediction = model(inputs)

    assert prediction.shape == (batch_size, 2, 1, 8, 8)
    assert torch.isfinite(prediction).all()

    model.train()
    training_batch = {
        **inputs,
        "target": torch.rand(batch_size, 2, 1, 8, 8),
    }
    output = model.training_step(training_batch, 0)
    assert output.prediction.shape == (batch_size, 2, 1, 8, 8)
    assert torch.isfinite(output.prediction).all()
    assert torch.isfinite(output.loss)

    output.loss.backward()
    processor_gradient = model.processor.delta_head.weight.grad
    assert processor_gradient is not None
    assert torch.count_nonzero(processor_gradient).item() > 0
