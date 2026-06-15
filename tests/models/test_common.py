import pytest
import torch

from icenet_mp.models.common import (
    ConvBlockUpsample,
    ConvNormActUpsample,
    NormalisedFold,
)


class TestConvBlockUpsample:
    @pytest.mark.parametrize("kernel_size", [2, 3, 4, 5])
    @pytest.mark.parametrize("in_channels", [4, 16])
    @pytest.mark.parametrize(("height", "width"), [(8, 8), (12, 20)])
    def test_output_shape(
        self, kernel_size: int, in_channels: int, height: int, width: int
    ) -> None:
        layer = ConvBlockUpsample(in_channels, kernel_size=kernel_size)
        x = torch.zeros(1, in_channels, height, width)
        y = layer(x)
        assert y.shape == (1, in_channels // 2, height * 2, width * 2)


class TestConvNormActUpsample:
    @pytest.mark.parametrize("kernel_size", [2, 3, 4, 5])
    @pytest.mark.parametrize("in_channels", [4, 16])
    @pytest.mark.parametrize("out_channels", [5, 13])
    @pytest.mark.parametrize(("height", "width"), [(8, 8), (12, 20)])
    def test_output_shape(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
    ) -> None:
        layer = ConvNormActUpsample(in_channels, out_channels, kernel_size=kernel_size)
        x = torch.zeros(1, in_channels, height, width)
        y = layer(x)
        assert y.shape == (1, out_channels, height * 2, width * 2)


class TestNormalisedFold:
    @pytest.mark.parametrize("input_chw", [(4, 57, 67), (1, 60, 50)])
    @pytest.mark.parametrize("latent_hw", [(32, 32), (20, 10)])
    def test_overlap_handling(
        self, input_chw: tuple[int, int, int], latent_hw: tuple[int, int]
    ) -> None:
        input_ones = torch.ones(1, *input_chw)
        input_hw = input_chw[1:]
        unfold = torch.nn.Unfold(
            kernel_size=latent_hw,
            stride=latent_hw,
            padding=latent_hw,
        )
        fold = NormalisedFold(
            output_size=input_hw,
            kernel_size=latent_hw,
            stride=latent_hw,
            padding=latent_hw,
        )
        output = fold(unfold(input_ones))
        assert torch.allclose(output, input_ones)
