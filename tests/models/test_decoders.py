import pytest
import torch

from icenet_mp.models.decoders import (
    BaseDecoder,
    CNNDecoder,
    NaiveLinearDecoder,
    PiecewiseDecoder,
)
from icenet_mp.types import DataSpace


class TestDecoders:
    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    @pytest.mark.parametrize(
        "test_decoder_cls", ["CNNDecoder", "NaiveLinearDecoder", "PiecewiseDecoder"]
    )
    @pytest.mark.parametrize("test_latent_chw", [(128, 32, 32), (2, 200, 100)])
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 3, 5])
    @pytest.mark.parametrize("test_output_chw", [(4, 256, 256), (1, 100, 200)])
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_decoder_cls: str,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_output_chw: tuple[int, int, int],
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )
        output_space = DataSpace(
            name="output", channels=test_output_chw[0], shape=test_output_chw[1:]
        )
        decoder: BaseDecoder = {
            "CNNDecoder": CNNDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_forecast_steps=test_n_forecast_steps,
                n_layers=1,
            ),
            "NaiveLinearDecoder": NaiveLinearDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_forecast_steps=test_n_forecast_steps,
            ),
            "PiecewiseDecoder": PiecewiseDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_forecast_steps=test_n_forecast_steps,
            ),
        }[test_decoder_cls]
        result: torch.Tensor = decoder.rollout(
            torch.randn(
                test_batch_size,
                test_n_forecast_steps,
                latent_space.channels,
                *latent_space.shape,
            )
        )
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            output_space.channels,
            *output_space.shape,
        )


class TestCNNDecoder:
    @pytest.mark.parametrize("test_latent_chw", [(3, 32, 32), (5, 200, 100)])
    @pytest.mark.parametrize("test_n_layers", [1, 2, 5])
    def test_latent_shape_errors(
        self, test_latent_chw: tuple[int, int, int], test_n_layers: int
    ) -> None:
        test_n_forecast_steps = 1
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )
        output_space = DataSpace(name="output", shape=(256, 256), channels=4)
        with pytest.raises(
            ValueError,
            match=f"The number of input channels {test_latent_chw[0]} must be divisible by {2**test_n_layers}. Without this, it is not possible to apply {test_n_layers} convolutions.",
        ):
            CNNDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_forecast_steps=test_n_forecast_steps,
                n_layers=test_n_layers,
            )


class TestDecoderBounded:
    @pytest.mark.xfail(
        reason="Bounded output for random input is not always between 0 and 1.",
        strict=False,
    )
    @pytest.mark.parametrize(
        "test_decoder_cls", ["CNNDecoder", "NaiveLinearDecoder", "PiecewiseDecoder"]
    )
    def test_bounded_fixes_values_between_0_and_1(self, test_decoder_cls: str) -> None:
        test_batch_size = 1
        test_n_forecast_steps = 1
        # choose latent channels divisible by 2 for CNNDecoder with n_layers=1
        latent_space = DataSpace(name="latent", channels=4, shape=(8, 8))
        output_space = DataSpace(name="output", channels=1, shape=(16, 16))

        decoders = {
            "CNNDecoder": lambda bounded: CNNDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_forecast_steps=test_n_forecast_steps,
                n_layers=1,
                bounded=bounded,
            ),
            "NaiveLinearDecoder": lambda bounded: NaiveLinearDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_forecast_steps=test_n_forecast_steps,
                bounded=bounded,
            ),
            "PiecewiseDecoder": lambda bounded: PiecewiseDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_forecast_steps=test_n_forecast_steps,
                restrict_range="tanh" if bounded else "none",
            ),
        }

        decoder_bounded = decoders[test_decoder_cls](bounded=True)
        decoder_unbounded = decoders[test_decoder_cls](bounded=False)

        # Large input values so that unbounded decoder outputs likely falls outside [0, 1]
        extreme_input = torch.full(
            (
                test_batch_size,
                test_n_forecast_steps,
                latent_space.channels,
                *latent_space.shape,
            ),
            1e10,
            dtype=torch.float32,
        )

        with torch.no_grad():
            out_bounded = decoder_bounded.rollout(extreme_input)
            out_unbounded = decoder_unbounded.rollout(extreme_input)

        # bounded output must be within [0, 1]
        assert torch.all(out_bounded >= 0.0).item()
        assert torch.all(out_bounded <= 1.0).item()

        # unbounded output should (very likely) contain values outside [0, 1]
        assert torch.any((out_unbounded < 0.0) | (out_unbounded > 1.0)).item()


class TestPiecewiseDecoder:
    @pytest.mark.parametrize("test_patch_size", [(2, 2), (3, 3), (7, 3)])
    @pytest.mark.parametrize("test_output_chw", [(4, 37, 53), (1, 256, 256)])
    def test_decoding_gives_same_range_as_input(
        self,
        test_patch_size: tuple[int, int],
        test_output_chw: tuple[int, int, int],
    ) -> None:
        # Generate input and output spaces
        output_space = DataSpace(
            name="output", channels=test_output_chw[0], shape=test_output_chw[1:]
        )
        stride = [max(1, p // 2) for p in test_patch_size]
        n_patches = (
            (output_space.shape[0] + 2 * stride[0] - (test_patch_size[0] - 1) - 1)
            // stride[0]
            + 1
        ) * (
            (output_space.shape[1] + 2 * stride[1] - (test_patch_size[1] - 1) - 1)
            // stride[1]
            + 1
        )
        input_space = DataSpace(
            name="input", channels=test_output_chw[0] * n_patches, shape=test_patch_size
        )

        # Initialise decoder
        decoder = PiecewiseDecoder(
            data_space_in=input_space,
            data_space_out=output_space,
            n_conv_blocks=0,
            n_forecast_steps=1,
            restrict_range="none",
        )

        # Generate a sequentially increasing input tensor
        input_ntchw = torch.arange(
            1,
            input_space.channels * input_space.shape[0] * input_space.shape[1] + 1,
            dtype=torch.float32,
        ).reshape(1, 1, input_space.channels, *input_space.shape)
        input_min_val = input_ntchw.min().item()
        input_max_val = input_ntchw.max().item()

        # Rollout the decoder and check that the output values are in the same range as the input values
        latent_ntchw = decoder.rollout(input_ntchw)
        assert latent_ntchw.shape == (1, 1, *output_space.chw)
        assert torch.all(input_min_val < latent_ntchw)
        assert torch.all(latent_ntchw < input_max_val)
