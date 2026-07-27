import pytest
import torch

from icenet_mp.models.common import (
    ConvBlockUpsample,
    ConvNormActUpsample,
    NormalisedFold,
)
from icenet_mp.models.common.gated_attention import GatedAttention, GatedAttentionBlock


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


class TestGatedAttention:
    @pytest.mark.parametrize("dilation", [0, -1])
    def test_dilation_below_one_raises(self, dilation: int) -> None:
        with pytest.raises(ValueError, match=r"dilation\(.*\) must be at least 1."):
            GatedAttention(channels=4, kernel_size=5, dilation=dilation)

    @pytest.mark.parametrize(("kernel_size", "dilation"), [(1, 2), (2, 3)])
    def test_kernel_size_below_dilation_raises(
        self, kernel_size: int, dilation: int
    ) -> None:
        with pytest.raises(
            ValueError, match=r"kernel_size \(.*\) must be >= dilation \(.*\)."
        ):
            GatedAttention(channels=4, kernel_size=kernel_size, dilation=dilation)


class TestGatedAttentionBlock:
    @pytest.mark.parametrize("drop_path_prob", [-0.1, 1.1])
    def test_drop_path_prob_outside_unit_interval_raises(
        self, drop_path_prob: float
    ) -> None:
        with pytest.raises(
            ValueError, match=r"drop_path_prob\(.*\) must be between 0 and 1."
        ):
            GatedAttentionBlock(
                4,
                4,
                kernel_size=5,
                dilation=1,
                mlp_ratio=2.0,
                drop_path_prob=drop_path_prob,
                mlp_drop_prob=0.0,
            )

    @pytest.mark.parametrize("mlp_drop_prob", [-0.1, 1.1])
    def test_mlp_drop_prob_outside_unit_interval_raises(
        self, mlp_drop_prob: float
    ) -> None:
        with pytest.raises(
            ValueError, match=r"mlp_drop_prob\(.*\) must be between 0 and 1."
        ):
            GatedAttentionBlock(
                4,
                4,
                kernel_size=5,
                dilation=1,
                mlp_ratio=2.0,
                drop_path_prob=0.5,
                mlp_drop_prob=mlp_drop_prob,
            )


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
