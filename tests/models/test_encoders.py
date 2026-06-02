import pytest
import torch

from icenet_mp.models.encoders import (
    BaseEncoder,
    CNNEncoder,
    NaiveLinearEncoder,
    PiecewiseEncoder,
)
from icenet_mp.types import DataSpace


class TestEncoders:
    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    @pytest.mark.parametrize(
        "test_encoder_cls", ["CNNEncoder", "NaiveLinearEncoder", "PiecewiseEncoder"]
    )
    @pytest.mark.parametrize("test_input_chw", [(4, 512, 512), (1, 20, 200)])
    @pytest.mark.parametrize("test_latent_hw", [(32, 32), (40, 73)])
    @pytest.mark.parametrize("test_n_history_steps", [1, 3, 5])
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_encoder_cls: str,
        test_input_chw: tuple[int, int, int],
        test_latent_hw: tuple[int, int],
        test_n_history_steps: int,
    ) -> None:
        input_space = DataSpace(
            name="input", channels=test_input_chw[0], shape=test_input_chw[1:]
        )
        encoder: BaseEncoder = {
            "CNNEncoder": CNNEncoder(
                data_space_in=input_space,
                latent_space=test_latent_hw,
                n_history_steps=test_n_history_steps,
            ),
            "NaiveLinearEncoder": NaiveLinearEncoder(
                data_space_in=input_space,
                latent_space=test_latent_hw,
                n_history_steps=test_n_history_steps,
            ),
            "PiecewiseEncoder": PiecewiseEncoder(
                data_space_in=input_space,
                latent_space=test_latent_hw,
                n_history_steps=test_n_history_steps,
            ),
        }[test_encoder_cls]
        result: torch.Tensor = encoder.rollout(
            torch.randn(
                test_batch_size,
                test_n_history_steps,
                input_space.channels,
                *input_space.shape,
            )
        )
        assert result.shape == (
            test_batch_size,
            test_n_history_steps,
            encoder.data_space_out.channels,
            *test_latent_hw,
        )


class TestPiecewiseEncoder:
    @pytest.mark.parametrize("test_input_chw", [(4, 512, 512), (1, 20, 200)])
    @pytest.mark.parametrize("test_latent_hw", [(32, 32), (40, 73)])
    def test_ones_are_encoded_to_zero_or_one(
        self,
        test_input_chw: tuple[int, int, int],
        test_latent_hw: tuple[int, int],
    ) -> None:
        input_space = DataSpace(
            name="input", channels=test_input_chw[0], shape=test_input_chw[1:]
        )
        encoder = PiecewiseEncoder(
            data_space_in=input_space,
            latent_space=test_latent_hw,
            n_conv_blocks=0,
            n_history_steps=1,
        )

        # Encode an input of ones
        input_ntchw = torch.ones((1, 1, *input_space.chw))
        latent_ntchw = encoder.rollout(input_ntchw)

        # Assert that the output only contains 0s and 1s, and that there are some of each
        assert torch.all(torch.isin(latent_ntchw, torch.tensor([0, 1])))
        assert torch.count_nonzero(1 - latent_ntchw) > 0  # count 0s
        assert torch.count_nonzero(latent_ntchw) > 0  # count 1s

    @pytest.mark.parametrize(
        "test_patches",
        [
            (
                (2, 2),
                [(1, 2, 6, 7), (2, 3, 7, 8), (7, 8, 12, 13)],
            ),
            (
                (3, 3),
                [(1, 2, 3, 6, 7, 8, 11, 12, 13), (7, 8, 9, 12, 13, 14, 17, 18, 19)],
            ),
        ],
    )
    def test_patches_are_extracted(
        self, test_patches: tuple[tuple[int, int], list[tuple[int, int, int, int]]]
    ) -> None:
        # Input is a 5x5 grid of values from 1 to 25
        input_ntchw = (
            torch.tensor(range(1, 26)).reshape(1, 1, 1, 5, 5).to(dtype=torch.float)
        )
        input_space = DataSpace(
            name="input", channels=input_ntchw.shape[2], shape=input_ntchw.shape[3:]
        )
        patch_shape: tuple[int, int] = test_patches[0]
        encoder = PiecewiseEncoder(
            data_space_in=input_space,
            latent_space=patch_shape,
            n_conv_blocks=0,
            n_history_steps=1,
        )
        latent_ntchw = encoder.rollout(input_ntchw)

        # Extract the p x p patches from the latent space and check that they match the expected patches
        latent_ntchw_patches = [
            latent_ntchw[0, 0, idx] for idx in range(latent_ntchw.shape[2])
        ]
        for expected_values in test_patches[1]:
            expected_patch = (
                torch.tensor(expected_values).reshape(patch_shape).to(dtype=torch.float)
            )
            # One of the latent patches should be the expected patch
            assert any(
                torch.allclose(expected_patch, latent_ntchw_patch)
                for latent_ntchw_patch in latent_ntchw_patches
            )
