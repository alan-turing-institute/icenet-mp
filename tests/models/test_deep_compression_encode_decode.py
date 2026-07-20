import pytest
import torch

from icenet_mp.models.decoders import DeepCompressionDecoder
from icenet_mp.models.encoders import DeepCompressionEncoder
from icenet_mp.types import DataSpace


class TestDeepCompressionEncoder:
    @pytest.mark.parametrize("pixel_shuffle", [True, False])
    @pytest.mark.parametrize("patch_size", [1, 2])
    def test_forward_shape(self, *, pixel_shuffle: bool, patch_size: int) -> None:
        stride = 2
        hid_channels = [4, 8, 16]
        hid_blocks = [1, 1, 1]
        spatial_factor = patch_size * stride ** (len(hid_channels) - 1)
        input_hw = spatial_factor * 4
        input_space = DataSpace(name="input", channels=3, shape=(input_hw, input_hw))

        encoder = DeepCompressionEncoder(
            data_space_in=input_space,
            latent_space=(4, 4),
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            patch_size=patch_size,
            pixel_shuffle=pixel_shuffle,
            stride=stride,
        )
        result = encoder.rollout(torch.randn(2, 3, *input_space.chw))
        assert result.shape == (2, 3, encoder.data_space_out.channels, 4, 4)


class TestDeepCompressionDecoder:
    @pytest.mark.parametrize("pixel_shuffle", [True, False])
    @pytest.mark.parametrize("patch_size", [1, 2])
    def test_forward_shape(self, *, pixel_shuffle: bool, patch_size: int) -> None:
        stride = 2
        hid_channels = [16, 8, 4]
        hid_blocks = [1, 1, 1]
        spatial_factor = patch_size * stride ** (len(hid_channels) - 1)
        output_hw = spatial_factor * 4
        latent_space = DataSpace(name="latent", channels=16, shape=(4, 4))
        output_space = DataSpace(
            name="output", channels=2, shape=(output_hw, output_hw)
        )

        decoder = DeepCompressionDecoder(
            data_space_in=latent_space,
            data_space_out=output_space,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            patch_size=patch_size,
            pixel_shuffle=pixel_shuffle,
            stride=stride,
        )
        result = decoder.rollout(torch.randn(2, 3, *latent_space.chw))
        assert result.shape == (2, 3, *output_space.chw)


class TestDeepCompressionShippedConfig:
    """Regression coverage for `icenet_mp/config/model/dc_unet_dc.yaml`.

    That config uses `pixel_shuffle: false` throughout, and its decoder's input
    channel count is the SUM of four independently-configured encoders' latent
    channels (96 + 8 + 2 + 16 = 122) against `hid_channels: [64, 128]` — not a clean
    multiple either way. This is exactly the case the residual shortcut's channel
    adaptation must handle without raising.
    """

    def test_era5_encoder_matches_shipped_config(self) -> None:
        input_space = DataSpace(name="era5", channels=7, shape=(432, 432))
        encoder = DeepCompressionEncoder(
            data_space_in=input_space,
            latent_space=(144, 144),
            hid_channels=[96, 192],
            hid_blocks=[2, 2],
            latent_channels=96,
            pixel_shuffle=False,
            stride=3,
        )
        result = encoder.rollout(torch.randn(1, 1, 7, 432, 432))
        assert result.shape == (1, 1, 96, 144, 144)

    def test_decoder_handles_non_power_of_two_combined_latent_channels(self) -> None:
        combined_latent_channels = (
            96 + 8 + 2 + 16
        )  # era5 + float-argo + sic-icenet + sic-ssmis
        latent_space = DataSpace(
            name="latent", channels=combined_latent_channels, shape=(144, 144)
        )
        output_space = DataSpace(name="sic-icenet", channels=1, shape=(432, 432))
        decoder = DeepCompressionDecoder(
            data_space_in=latent_space,
            data_space_out=output_space,
            hid_channels=[64, 128],
            hid_blocks=[2, 2],
            pixel_shuffle=False,
            stride=3,
        )
        result = decoder.rollout(torch.randn(1, 1, combined_latent_channels, 144, 144))
        assert result.shape == (1, 1, 1, 432, 432)
