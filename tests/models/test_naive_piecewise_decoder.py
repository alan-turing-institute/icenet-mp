from importlib.resources import files

import torch
import yaml

from icenet_mp.models.decoders import PiecewiseDecoder
from icenet_mp.types import DataSpace


def _make_decoder() -> PiecewiseDecoder:
    return PiecewiseDecoder(
        data_space_in=DataSpace(name="combined", channels=90, shape=(4, 4)),
        data_space_out=DataSpace(name="sic", channels=1, shape=(8, 8)),
        target_channel_offset=10,
        target_group_channels=3,
        target_variable_indices=[1],
        conv_subblocks_initial=0,
        conv_subblocks_final=0,
        use_final_normalisation=False,
        use_hann_window=False,
    )


def test_naive_piecewise_decoder_selects_target_variable_from_each_patch() -> None:
    decoder = _make_decoder()

    assert decoder.input_channel_indices == tuple(
        10 + patch_idx * 3 + 1 for patch_idx in range(25)
    )


def test_naive_piecewise_decoder_has_no_convolutions_and_returns_output_shape() -> None:
    decoder = _make_decoder()
    x = torch.randn(2, 90, 4, 4)

    output = decoder(x)

    assert output.shape == (2, 1, 8, 8)
    assert not any(isinstance(module, torch.nn.Conv2d) for module in decoder.modules())


def test_naive_piecewise_decoder_requires_target_layout_metadata() -> None:
    try:
        PiecewiseDecoder(
            data_space_in=DataSpace(name="combined", channels=90, shape=(4, 4)),
            data_space_out=DataSpace(name="sic", channels=1, shape=(8, 8)),
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            use_final_normalisation=False,
        )
    except ValueError as exc:
        assert "target latent-channel metadata" in str(exc)
    else:
        raise AssertionError("Expected target-layout validation to fail")


def test_naive_piecewise_model_config_disables_encoder_and_decoder_convolutions() -> None:
    config_path = files("icenet_mp.config") / "model" / (
        "piecewise_unet_piecewise_naive.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    for name, encoder in config["encoders"].items():
        if name == "latent_space":
            continue
        assert encoder["conv_subblocks_initial"] == 0
        assert encoder["conv_subblocks_final"] == 0

    decoder = config["decoder"]
    assert decoder["conv_subblocks_initial"] == 0
    assert decoder["conv_subblocks_final"] == 0
