import pytest
import torch
from torch import nn

from icenet_mp.models.common import (
    ChannelAdaptor,
    ConvBlockUpsample,
    ConvNormActUpsample,
    NormalisedFold,
    ResidualDownsample,
    ResidualUpsample,
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


class TestChannelAdapt:
    def test_identity_when_equal(self) -> None:
        x = torch.randn(2, 5, 3, 3)
        assert torch.equal(ChannelAdaptor(5, 5)(x), x)

    def test_shrink_exact_multiple_averages_contiguous_groups(self) -> None:
        # channel c holds the constant value c; groups of 4 average to 1.5 and 5.5
        x = torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1).expand(1, 8, 2, 2)
        y = ChannelAdaptor(8, 2)(x)
        assert y.shape == (1, 2, 2, 2)
        assert torch.allclose(y[:, 0], torch.full((1, 2, 2), 1.5))
        assert torch.allclose(y[:, 1], torch.full((1, 2, 2), 5.5))

    def test_grow_exact_multiple_duplicates_contiguous_groups(self) -> None:
        x = torch.tensor([10.0, 20.0]).view(1, 2, 1, 1).expand(1, 2, 2, 2)
        y = ChannelAdaptor(2, 8)(x)
        assert y.shape == (1, 8, 2, 2)
        assert torch.allclose(y[:, :4], torch.full((1, 4, 2, 2), 10.0))
        assert torch.allclose(y[:, 4:], torch.full((1, 4, 2, 2), 20.0))

    @pytest.mark.parametrize(("in_channels", "out_channels"), [(7, 3), (3, 7), (5, 5)])
    def test_non_exact_ratio_produces_correct_shape(
        self, in_channels: int, out_channels: int
    ) -> None:
        # Non-divisible ratios (e.g. a multi-dataset combined latent) must not raise.
        x = torch.randn(2, in_channels, 4, 4)
        y = ChannelAdaptor(in_channels, out_channels)(x)
        assert y.shape == (2, out_channels, 4, 4)


def _zero_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.data.zero_()


class TestResidualDownsample:
    @pytest.mark.parametrize("pixel_shuffle", [True, False])
    def test_zeroed_parametric_leaves_only_the_shortcut(
        self, *, pixel_shuffle: bool
    ) -> None:
        in_channels, out_channels, factor = 4, 8, 2
        block = ResidualDownsample(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            pixel_shuffle=pixel_shuffle,
            kernel_size=1,
        )
        _zero_params(block.parametric)

        x = torch.randn(2, in_channels, 8, 8)
        assert torch.allclose(block(x), block.shortcut(x))
        assert block(x).shape == (2, out_channels, 4, 4)


class TestResidualUpsample:
    @pytest.mark.parametrize("pixel_shuffle", [True, False])
    def test_zeroed_parametric_leaves_only_the_shortcut(
        self, *, pixel_shuffle: bool
    ) -> None:
        in_channels, out_channels, factor = 8, 4, 2
        block = ResidualUpsample(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            pixel_shuffle=pixel_shuffle,
            kernel_size=1,
        )
        _zero_params(block.parametric)

        x = torch.randn(2, in_channels, 4, 4)
        assert torch.allclose(block(x), block.shortcut(x))
        assert block(x).shape == (2, out_channels, 8, 8)


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
