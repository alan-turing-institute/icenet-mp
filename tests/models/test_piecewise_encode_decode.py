import pytest
import torch

from icenet_mp.models.decoders import PiecewiseDecoder
from icenet_mp.models.encoders import PiecewiseEncoder
from icenet_mp.types import DataSpace


class TestPiecewiseEncodeDecode:
    @pytest.mark.parametrize("test_batch_size", [1, 2, 3])
    @pytest.mark.parametrize("test_input_chw", [(1, 512, 512), (4, 20, 60)])
    @pytest.mark.parametrize("test_patch_size", [(2, 2), (4, 2), (5, 3)])
    @pytest.mark.parametrize("test_timesteps", [1, 2, 3])
    def test_forward(
        self,
        test_batch_size: int,
        test_input_chw: tuple[int, int, int],
        test_patch_size: tuple[int, int],
        test_timesteps: int,
    ) -> None:
        # In order to exactly reproduce the input, we need:
        # - timesteps to be the same in the encoder and decoder
        n_history_steps = n_forecast_steps = test_timesteps
        # - patch size to divide the input size
        if (
            test_input_chw[1] % test_patch_size[0] != 0
            or test_input_chw[2] % test_patch_size[1] != 0
        ):
            pytest.skip("Patch size must divide the input size for this test.")
        # - no convolutional blocks, to avoid changing the values
        n_conv_blocks = 0
        input_ntchw = (
            torch.tensor(
                range(
                    1,
                    test_batch_size
                    * n_history_steps
                    * test_input_chw[0]
                    * test_input_chw[1]
                    * test_input_chw[2]
                    + 1,
                )
            )
            .reshape(
                test_batch_size,
                n_history_steps,
                test_input_chw[0],
                test_input_chw[1],
                test_input_chw[2],
            )
            .to(dtype=torch.float)
        )
        input_space = DataSpace(
            name="input", channels=input_ntchw.shape[2], shape=input_ntchw.shape[3:]
        )
        encoder = PiecewiseEncoder(
            data_space_in=input_space,
            latent_space=test_patch_size,
            n_conv_blocks=n_conv_blocks,
            n_history_steps=n_history_steps,
        )
        latent_ntchw = encoder.rollout(input_ntchw)
        decoder = PiecewiseDecoder(
            data_space_in=encoder.data_space_out,
            data_space_out=input_space,
            n_conv_blocks=n_conv_blocks,
            n_forecast_steps=n_forecast_steps,
        )
        output_ntchw = decoder.rollout(latent_ntchw)
        assert torch.equal(input_ntchw, output_ntchw)
