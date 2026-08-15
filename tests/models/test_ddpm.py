import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from icenet_mp.models.ddpm import DDPM, SimpleEncoder2D


def _make_ddpm(*, use_autoregressive: bool = True, **kwargs) -> DDPM:  # noqa: ANN003
    input_spaces = [
        DictConfig(
            {
                "channels": 1,
                "name": "osisaf-north",
                "shape": (16, 16),
            }
        ),
        DictConfig(
            {
                "channels": 2,
                "name": "era5",
                "shape": (8, 8),
            }
        ),
    ]
    output_space = DictConfig(
        {
            "channels": 1,
            "name": "osisaf-north",
            "shape": (16, 16),
        }
    )
    model_kwargs = {
        "hemisphere": "north",
        "input_spaces": input_spaces,
        "loss": OmegaConf.create({"_target_": "torch.nn.HuberLoss"}),
        "metrics": [],
        "n_forecast_steps": 3,
        "n_history_steps": 2,
        "name": "ddpm-test",
        "optimizer": DictConfig({}),
        "output_space": output_space,
        "scheduler": DictConfig({}),
        "start_out_channels": 4,
        "time_embed_dim": 16,
        "timesteps": 2,
        "use_autoregressive": use_autoregressive,
    }
    model_kwargs.update(kwargs)
    return DDPM(**model_kwargs)


class TestSimpleEncoder2D:
    def test_forward_shape(self) -> None:
        encoder = SimpleEncoder2D(in_channels=3, out_channels=8)
        result = encoder(torch.randn(2, 3, 16, 16))
        assert result.shape == (2, 8, 16, 16)


class TestDDPM:
    @pytest.mark.parametrize(
        ("use_autoregressive", "expected_output_channels"),
        [(True, 1), (False, 3)],
    )
    def test_output_channels_follow_sampling_mode(
        self, use_autoregressive: bool, expected_output_channels: int
    ) -> None:
        model = _make_ddpm(use_autoregressive=use_autoregressive)
        assert model.output_channels == expected_output_channels

    def test_prepare_inputs_resizes_and_combines_conditioning(self) -> None:
        model = _make_ddpm()
        batch = {
            "osisaf-north": torch.randn(2, 2, 1, 16, 16),
            "era5": torch.randn(2, 2, 2, 8, 8),
        }

        result = model.prepare_inputs(batch)

        assert result.shape == (2, model.cond_channels, 16, 16)

    def test_forward_is_not_supported(self) -> None:
        model = _make_ddpm()
        with pytest.raises(NotImplementedError, match="training_step"):
            model.forward()

    @pytest.mark.parametrize(
        ("argument", "value", "message"),
        [
            ("kernel_size", 0, "Kernel size must be greater than 0"),
            ("start_out_channels", 0, "Start out channels must be greater than 0"),
        ],
    )
    def test_invalid_unet_arguments_raise(
        self, argument: str, value: int, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _make_ddpm(**{argument: value})
