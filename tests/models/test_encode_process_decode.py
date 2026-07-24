import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.models import EncodeProcessDecode


@pytest.mark.parametrize("test_n_forecast_steps", [1, 2, 5])
@pytest.mark.parametrize("test_n_history_steps", [1, 2, 5])
class TestEncodeProcessDecode:
    def test_init(  # noqa: PLR0917
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            loss=cfg_loss,
            target_variable_indices=[0],
        )

        assert model.name == "encode-null-decode"
        assert model.input_spaces[0].channels == cfg_input_space["channels"]
        assert model.input_spaces[0].name == cfg_input_space["name"]
        assert model.input_spaces[0].shape == cfg_input_space["shape"]
        assert model.n_forecast_steps == test_n_forecast_steps
        assert model.n_history_steps == test_n_history_steps
        assert model.output_space.channels == cfg_output_space["channels"]
        assert model.output_space.name == cfg_output_space["name"]
        assert model.output_space.shape == cfg_output_space["shape"]

    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    def test_forward(  # noqa: PLR0917
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            loss=cfg_loss,
            target_variable_indices=[0],
        )
        result: torch.Tensor = model(
            {
                cfg_input_space["name"]: torch.randn(
                    test_batch_size,
                    test_n_history_steps,
                    cfg_input_space["channels"],
                    cfg_input_space["shape"][0],
                    cfg_input_space["shape"][1],
                ),
                cfg_output_space["name"]: torch.rand(
                    test_batch_size,
                    test_n_history_steps,
                    cfg_output_space["channels"],
                    cfg_output_space["shape"][0],
                    cfg_output_space["shape"][1],
                ),
            }
        )
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            cfg_output_space["channels"],
            cfg_output_space["shape"][0],
            cfg_output_space["shape"][1],
        )

    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    def test_forward_with_motion_channels(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            loss=cfg_loss,
            use_motion_channels=True,
        )
        history = torch.randn(
            test_batch_size,
            test_n_history_steps,
            cfg_input_space["channels"],
            cfg_input_space["shape"][0],
            cfg_input_space["shape"][1],
        )
        result: torch.Tensor = model({cfg_input_space["name"]: history})
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            cfg_output_space["channels"],
            cfg_output_space["shape"][0],
            cfg_output_space["shape"][1],
        )

        # The first history step has no earlier frame, so its motion channels must be
        # zero-filled rather than garbage/uninitialised.
        augmented = model._with_motion_channels(history)
        n_raw_channels = cfg_input_space["channels"]
        assert torch.equal(
            augmented[:, 0, n_raw_channels:], torch.zeros_like(history[:, 0])
        )
        if test_n_history_steps > 1:
            expected_diff = history[:, 1] - history[:, 0]
            assert torch.equal(augmented[:, 1, n_raw_channels:], expected_diff)

    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    def test_forward_with_day_order_channels(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            loss=cfg_loss,
            use_day_order_channels=True,
        )
        history = torch.randn(
            test_batch_size,
            test_n_history_steps,
            cfg_input_space["channels"],
            cfg_input_space["shape"][0],
            cfg_input_space["shape"][1],
        )
        result: torch.Tensor = model({cfg_input_space["name"]: history})
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            cfg_output_space["channels"],
            cfg_output_space["shape"][0],
            cfg_output_space["shape"][1],
        )

        # Exactly one label channel is appended per timestep, constant within a
        # timestep and evenly spaced from 0 (oldest) to 1 (most recent).
        augmented = model._with_day_order_channels(history)
        n_raw_channels = cfg_input_space["channels"]
        assert augmented.shape[2] == n_raw_channels + 1
        assert torch.equal(
            augmented[:, 0, n_raw_channels], torch.zeros_like(history[:, 0, 0])
        )
        if test_n_history_steps > 1:
            assert torch.equal(
                augmented[:, -1, n_raw_channels], torch.ones_like(history[:, 0, 0])
            )

    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    def test_forward_with_skip_connection(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            loss=cfg_loss,
            use_skip_connection=True,
        )
        result: torch.Tensor = model(
            {
                cfg_input_space["name"]: torch.randn(
                    test_batch_size,
                    test_n_history_steps,
                    cfg_input_space["channels"],
                    cfg_input_space["shape"][0],
                    cfg_input_space["shape"][1],
                )
            }
        )
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            cfg_output_space["channels"],
            cfg_output_space["shape"][0],
            cfg_output_space["shape"][1],
        )

    @pytest.mark.parametrize("test_batch_size", [1, 2])
    def test_forward_with_residual_output(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        # The target group must be an input dataset in residual mode, so build an
        # input space with the target's name and more channels than the output.
        input_space = DictConfig(
            {
                "channels": 4,
                "name": cfg_output_space["name"],
                "shape": cfg_output_space["shape"],
            }
        )
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            loss=cfg_loss,
            use_residual_output=True,
            target_variable_indices=[2],
        )
        history = torch.rand(
            test_batch_size,
            test_n_history_steps,
            input_space["channels"],
            input_space["shape"][0],
            input_space["shape"][1],
        )
        result: torch.Tensor = model({input_space["name"]: history})
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            cfg_output_space["channels"],
            cfg_output_space["shape"][0],
            cfg_output_space["shape"][1],
        )
        assert result.min() >= 0.0
        assert result.max() <= 1.0

        # With every decoder parameter zeroed the delta is zero, so the model must
        # reproduce persistence exactly: the last observed frame of the target
        # channel, repeated across all forecast steps.
        for parameter in model.decoder.parameters():
            parameter.data.zero_()
        result = model({input_space["name"]: history})
        expected = history[:, -1:, [2], :, :].expand(
            -1, test_n_forecast_steps, -1, -1, -1
        )
        assert torch.allclose(result, expected)

    def test_residual_output_requires_target_input(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        # The default fixtures use different names for input ("test-input") and
        # output ("target") spaces, so residual mode must refuse to build.
        with pytest.raises(ValueError, match="use_residual_output"):
            EncodeProcessDecode(
                name="encode-null-decode",
                encoders=cfg_encoders,
                processor=cfg_processor,
                decoder=cfg_decoder,
                hemisphere="north",
                input_spaces=[cfg_input_space],
                n_forecast_steps=test_n_forecast_steps,
                n_history_steps=test_n_history_steps,
                output_space=cfg_output_space,
                optimizer=DictConfig({}),
                scheduler=DictConfig({}),
                loss=cfg_loss,
                use_residual_output=True,
            )
