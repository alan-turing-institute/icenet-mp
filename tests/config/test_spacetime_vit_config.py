from importlib.resources import files

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def test_spacetime_vit_model_config_enables_missing_argo_strategy() -> None:
    config_dir = str(files("icenet_mp.config"))
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(
                config_name="sample",
                overrides=["model=cnn_spacetime_vit_cnn"],
            )
    finally:
        GlobalHydra.instance().clear()

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
