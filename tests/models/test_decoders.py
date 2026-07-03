from pathlib import Path

import numpy as np
import pytest
import torch

from icenet_mp.models.decoders import (
    BaseDecoder,
    CNNDecoder,
    NaiveLinearDecoder,
    PiecewiseDecoder,
)
from icenet_mp.types import DataSpace

# Real SSMIS active grid cell mask, copied locally. Absent in CI -> the integration
# test below is skipped there and only runs where the sample data exists.
_REAL_MASKS = sorted(
    (Path(__file__).resolve().parents[2] / "my_data").glob("**/active_mask.npy")
)


class TestDecoders:
    @pytest.mark.parametrize("test_batch_size", [1, 2])
    @pytest.mark.parametrize(
        "test_decoder_cls", ["CNNDecoder", "NaiveLinearDecoder", "PiecewiseDecoder"]
    )
    @pytest.mark.parametrize("test_latent_chw", [(128, 32, 32), (2, 200, 100)])
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 5])
    @pytest.mark.parametrize("test_output_chw", [(4, 64, 64), (1, 20, 20)])
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
                n_layers=1,
            ),
            "NaiveLinearDecoder": NaiveLinearDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
            ),
            "PiecewiseDecoder": PiecewiseDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
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
                n_layers=test_n_layers,
            )


class TestDecoderBounded:
    @pytest.mark.parametrize(
        "test_decoder_cls", ["CNNDecoder", "NaiveLinearDecoder", "PiecewiseDecoder"]
    )
    def test_bounded_fixes_values_between_0_and_1(self, test_decoder_cls: str) -> None:
        test_n_forecast_steps = 1
        # latent channels must be divisible by 2 for CNNDecoder with n_layers=1
        latent_space = DataSpace(name="latent", channels=4, shape=(8, 8))
        output_space = DataSpace(name="output", channels=1, shape=(16, 16))

        decoder = {
            "CNNDecoder": CNNDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_layers=1,
                restrict_range="sigmoid",
            ),
            "NaiveLinearDecoder": NaiveLinearDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                restrict_range="sigmoid",
            ),
            "PiecewiseDecoder": PiecewiseDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                restrict_range="tanh",
            ),
        }[test_decoder_cls]

        extreme_input = torch.full(
            (1, test_n_forecast_steps, latent_space.channels, *latent_space.shape),
            1e10,
            dtype=torch.float32,
        )
        with torch.no_grad():
            out = decoder.rollout(extreme_input)

        assert torch.all(out >= 0.0).item()
        assert torch.all(out <= 1.0).item()


class TestDecoderMask:
    """The active grid cell mask, applied via BaseDecoder.finalise()."""

    @staticmethod
    def _spaces() -> tuple[DataSpace, DataSpace]:
        latent_space = DataSpace(name="latent", channels=4, shape=(8, 8))
        output_space = DataSpace(name="output", channels=1, shape=(16, 16))
        return latent_space, output_space

    def test_use_mask_loads_and_zeros_masked_cells(self, tmp_path) -> None:  # noqa: ANN001
        latent_space, output_space = self._spaces()
        # Mask the top half (0 = inactive), keep the bottom half (1 = active).
        mask = np.ones(output_space.shape, dtype=np.uint8)
        mask[:8, :] = 0
        mask_path = tmp_path / "active_mask.npy"
        np.save(mask_path, mask)

        decoder = NaiveLinearDecoder(
            active_mask_path=str(mask_path),
            data_space_in=latent_space,
            data_space_out=output_space,
            mask_type="active",
        )

        assert decoder.mask.shape == output_space.shape
        out = decoder.rollout(
            torch.randn(2, 1, latent_space.channels, *latent_space.shape)
        )
        # Every masked (inactive) cell must be exactly zero; active cells are untouched.
        assert torch.all(out[..., :8, :] == 0).item()

    def test_use_mask_without_file_raises(self, tmp_path) -> None:  # noqa: ANN001
        latent_space, output_space = self._spaces()
        with pytest.raises(FileNotFoundError, match="mask is requested"):
            NaiveLinearDecoder(
                active_mask_path=str(tmp_path / "does_not_exist.npy"),
                data_space_in=latent_space,
                data_space_out=output_space,
                mask_type="active",
            )

    def test_land_mask_loads_and_zeros_masked_cells(self, tmp_path) -> None:  # noqa: ANN001
        latent_space, output_space = self._spaces()
        # Land = 0 (top half), sea = 1 (bottom half).
        mask = np.ones(output_space.shape, dtype=np.uint8)
        mask[:8, :] = 0
        mask_path = tmp_path / "land_mask.npy"
        np.save(mask_path, mask)

        decoder = NaiveLinearDecoder(
            data_space_in=latent_space,
            data_space_out=output_space,
            land_mask_path=str(mask_path),
            mask_type="land",
        )

        assert decoder.mask.shape == output_space.shape
        out = decoder.rollout(
            torch.randn(2, 1, latent_space.channels, *latent_space.shape)
        )
        # Land cells exactly zero; sea cells (incl. confident-no-ice) left free.
        assert torch.all(out[..., :8, :] == 0).item()

    def test_mask_type_none_creates_no_buffer(self) -> None:
        """An explicit mask_type='none' behaves like no mask: no buffer, no multiply."""
        latent_space, output_space = self._spaces()
        decoder = NaiveLinearDecoder(
            data_space_in=latent_space,
            data_space_out=output_space,
            mask_type="none",
        )
        assert decoder.use_mask is False
        assert not hasattr(decoder, "mask")

    def test_unknown_mask_type_raises(self) -> None:
        """A typo'd mask_type fails loudly rather than silently disabling masking."""
        latent_space, output_space = self._spaces()
        with pytest.raises(ValueError, match="Unknown mask_type"):
            NaiveLinearDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                mask_type="activ",
            )

    def test_mask_off_creates_no_buffer_and_skips_multiply(self) -> None:
        latent_space, output_space = self._spaces()
        decoder = NaiveLinearDecoder(
            data_space_in=latent_space,
            data_space_out=output_space,
        )
        # No mask requested: no dummy buffer is created and finalise() skips the
        # multiply entirely (rather than doing an identity product with ones).
        assert decoder.use_mask is False
        assert not hasattr(decoder, "mask")

    def test_use_mask_with_bounded_keeps_masked_cells_exactly_zero(
        self, tmp_path: Path
    ) -> None:
        """Masking is applied AFTER sigmoid, so masked cells are 0, not sigmoid(0)=0.5."""
        latent_space, output_space = self._spaces()
        mask = np.ones(output_space.shape, dtype=np.uint8)
        mask[:8, :] = 0  # top half inactive
        mask_path = tmp_path / "active_mask.npy"
        np.save(mask_path, mask)

        decoder = NaiveLinearDecoder(
            active_mask_path=str(mask_path),
            data_space_in=latent_space,
            data_space_out=output_space,
            mask_type="active",
            restrict_range="sigmoid",
        )
        out = decoder.rollout(
            torch.randn(2, 1, latent_space.channels, *latent_space.shape)
        )
        # Masked cells must be exactly 0 even with bounding on...
        assert torch.all(out[..., :8, :] == 0).item()
        # ...and active cells are bounded to [0, 1] by the sigmoid.
        active = out[..., 8:, :]
        assert torch.all((active >= 0) & (active <= 1)).item()

    def test_finalise_is_identity_when_mask_and_bound_off(self) -> None:
        """Backward compatibility: with both off, finalise() must not touch the tensor."""
        latent_space, output_space = self._spaces()
        decoder = NaiveLinearDecoder(
            data_space_in=latent_space,
            data_space_out=output_space,
        )
        x = torch.randn(2, output_space.channels, *output_space.shape)
        assert torch.equal(decoder.finalise(x), x)


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

    def test_clamp_restricts_output_to_unit_range(self) -> None:
        output_space = DataSpace(name="output", channels=1, shape=(4, 4))
        patch_size = (2, 2)
        stride = [max(1, p // 2) for p in patch_size]
        n_patches = (
            (output_space.shape[0] + 2 * stride[0] - (patch_size[0] - 1) - 1)
            // stride[0]
            + 1
        ) * (
            (output_space.shape[1] + 2 * stride[1] - (patch_size[1] - 1) - 1)
            // stride[1]
            + 1
        )
        input_space = DataSpace(name="input", channels=n_patches, shape=patch_size)
        decoder = PiecewiseDecoder(
            data_space_in=input_space,
            data_space_out=output_space,
            n_conv_blocks=0,
            restrict_range="clamp",
        )
        x = torch.full(
            (1, 1, input_space.channels, *input_space.shape), 1e10, dtype=torch.float32
        )
        output = decoder.rollout(x)
        assert torch.all(output >= 0.0).item()
        assert torch.all(output <= 1.0).item()


@pytest.mark.skipif(
    not _REAL_MASKS, reason="No real SSMIS active mask available locally"
)
class TestDecoderMaskOnRealMask:
    """Integration check against the real 432x432 active grid cell mask on disk."""

    def test_decoder_zeros_exactly_the_inactive_cells(self) -> None:
        """The decoder output is 0 on exactly the masked cells, nothing else.

        Stronger than a visual check: across several independent random inputs, the
        cells that are *always* zero must be exactly the mask's inactive cells.
        """
        mask_np = np.load(_REAL_MASKS[0])
        output_space = DataSpace(name="sic", channels=1, shape=tuple(mask_np.shape))
        latent_space = DataSpace(name="latent", channels=8, shape=(108, 108))
        decoder = NaiveLinearDecoder(
            active_mask_path=str(_REAL_MASKS[0]),
            data_space_in=latent_space,
            data_space_out=output_space,
            n_forecast_steps=1,
            mask_type="active",
        )

        inactive = torch.from_numpy(mask_np == 0)
        # The decoder loaded exactly the on-disk mask.
        assert torch.equal(decoder.mask.bool(), torch.from_numpy(mask_np != 0))

        # Always_zero starts all-True and can only shrink as samples fire active cells,
        # converging down to inactive. An active cell that is 0 by chance in the first
        # few draws is cleared by a later one; an active cell that is structurally always
        # 0 never clears. So draw until it converges (50 repetition is used here), and
        # fail if it never does. This should better differentiate a rare coincidence (clears fast)
        # from a genuine bug (persists no matter how many draws).
        always_zero = torch.ones_like(inactive)
        for seed in range(50):
            torch.manual_seed(seed)
            out = decoder.rollout(
                torch.randn(2, 1, latent_space.channels, *latent_space.shape)
            )
            # out is (N, n_forecast, C=1, H, W); reduce all but the spatial dims
            always_zero &= (out == 0).all(dim=(0, 1, 2))
            if torch.equal(always_zero, inactive):
                break

        assert torch.equal(always_zero, inactive), (
            "Note: we can safely assume an active cell stays zero across all 50 draws is a real over-zeroing bug, "
            "not the extremely rare coincidence."
        )
        # only use none trivial mask for the test (genuinely some active and some inactive cells).
        assert inactive.any().item()
        assert (~inactive).any().item()
