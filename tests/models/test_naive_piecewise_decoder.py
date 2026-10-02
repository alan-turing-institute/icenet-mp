from importlib.resources import files

import pytest
import torch
import yaml

from icenet_mp.models.decoders import PiecewiseDecoder
from icenet_mp.models.encoders import PiecewiseEncoder
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
    """Select the requested target variable from every target-group patch."""
    decoder = _make_decoder()

    assert decoder.input_channel_indices == tuple(
        10 + patch_idx * 3 + 1 for patch_idx in range(25)
    )


def test_naive_piecewise_round_trip_selects_target_variable() -> None:
    """Recover the selected variable from a target encoder inside a combined latent."""
    target_space = DataSpace(name="sic", channels=3, shape=(8, 8))
    encoder = PiecewiseEncoder(
        data_space_in=target_space,
        latent_space=(4, 4),
        conv_subblocks_initial=0,
        conv_subblocks_final=0,
    )
    source = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    target_latent = encoder(source)

    # Represent a preceding one-channel piecewise encoder: 25 patches precede the
    # target group's 75 channels in the combined latent.
    prefix = torch.zeros(1, 25, 4, 4)
    combined_latent = torch.cat((prefix, target_latent), dim=1)
    decoder = PiecewiseDecoder(
        data_space_in=DataSpace(
            name="combined", channels=combined_latent.shape[1], shape=(4, 4)
        ),
        data_space_out=DataSpace(name="sic", channels=1, shape=(8, 8)),
        target_channel_offset=25,
        target_group_channels=3,
        target_variable_indices=[1],
        conv_subblocks_initial=0,
        conv_subblocks_final=0,
        use_final_normalisation=False,
        use_hann_window=False,
    )

    output = decoder(combined_latent)

    torch.testing.assert_close(output, source[:, 1:2])


def test_naive_piecewise_decoder_has_no_convolutions_and_returns_output_shape() -> None:
    """Decode target patches without introducing learned convolutions."""
    decoder = _make_decoder()
    x = torch.randn(2, 90, 4, 4)

    output = decoder(x)

    assert output.shape == (2, 1, 8, 8)
    assert not any(isinstance(module, torch.nn.Conv2d) for module in decoder.modules())


def test_naive_piecewise_decoder_requires_target_layout_metadata() -> None:
    """Require target layout metadata for multimodal convolution-free decoding."""
    with pytest.raises(ValueError, match="target latent-channel metadata"):
        PiecewiseDecoder(
            data_space_in=DataSpace(name="combined", channels=90, shape=(4, 4)),
            data_space_out=DataSpace(name="sic", channels=1, shape=(8, 8)),
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            use_final_normalisation=False,
        )


def test_naive_piecewise_decoder_requires_one_index_per_output_channel() -> None:
    """Reject target layouts that select the wrong number of output variables."""
    with pytest.raises(ValueError, match="Expected 1 target variable indices"):
        PiecewiseDecoder(
            data_space_in=DataSpace(name="combined", channels=90, shape=(4, 4)),
            data_space_out=DataSpace(name="sic", channels=1, shape=(8, 8)),
            target_channel_offset=10,
            target_group_channels=3,
            target_variable_indices=[0, 1],
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            use_final_normalisation=False,
        )


def test_naive_piecewise_decoder_rejects_invalid_target_variable_index() -> None:
    """Require selected variables to exist inside the target input group."""
    with pytest.raises(
        ValueError, match="must refer to channels in the target input group"
    ):
        PiecewiseDecoder(
            data_space_in=DataSpace(name="combined", channels=90, shape=(4, 4)),
            data_space_out=DataSpace(name="sic", channels=1, shape=(8, 8)),
            target_channel_offset=10,
            target_group_channels=3,
            target_variable_indices=[3],
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            use_final_normalisation=False,
        )


def test_naive_piecewise_decoder_rejects_out_of_bounds_target_layout() -> None:
    """Reject target patch layouts that extend beyond the combined latent channels."""
    with pytest.raises(ValueError, match="exceeds combined latent channels"):
        PiecewiseDecoder(
            data_space_in=DataSpace(name="combined", channels=90, shape=(4, 4)),
            data_space_out=DataSpace(name="sic", channels=1, shape=(8, 8)),
            target_channel_offset=20,
            target_group_channels=3,
            target_variable_indices=[1],
            conv_subblocks_initial=0,
            conv_subblocks_final=0,
            use_final_normalisation=False,
        )


def test_naive_piecewise_model_config_disables_encoder_and_decoder_convolutions() -> (
    None
):
    """Disable learned convolutions in every piecewise encoder and the decoder."""
    config_path = (
        files("icenet_mp.config") / "model" / ("piecewise_unet_piecewise_naive.yaml")
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
