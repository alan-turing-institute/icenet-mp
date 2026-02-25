import pytest
import torch

from icenet_mp.models.common import NormalisedFold


class TestNormalisedFold:
    @pytest.mark.parametrize("test_input_chw", [(4, 57, 67), (1, 60, 50)])
    @pytest.mark.parametrize("test_latent_hw", [(32, 32), (20, 10)])
    def test_overlap_handling(
        self, test_input_chw: tuple[int, int, int], test_latent_hw: tuple[int, int]
    ) -> None:
        input_ones = torch.ones(1, *test_input_chw)
        input_hw = test_input_chw[1:]
        unfold = torch.nn.Unfold(
            kernel_size=test_latent_hw,
            stride=test_latent_hw,
            padding=test_latent_hw,
        )
        fold = NormalisedFold(
            output_size=input_hw,
            kernel_size=test_latent_hw,
            stride=test_latent_hw,
            padding=test_latent_hw,
        )
        output = fold(unfold(input_ones))
        assert torch.allclose(output, input_ones)
