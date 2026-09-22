import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.models import EncodeProcessDecode


@pytest.mark.parametrize("test_n_forecast_steps", [1, 2, 5])
@pytest.mark.parametrize("test_n_history_steps", [1, 2, 5])
class TestEncodeProcessDecode:
    def test_init(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
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
            lr_scheduler=DictConfig({}),
            loss=cfg_loss,
            metrics=cfg_metrics,
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
    def test_forward(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
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
            loss=cfg_loss,
            metrics=cfg_metrics,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
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

    def test_processor_default_does_not_require_multistage(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
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
            loss=cfg_loss,
            metrics=cfg_metrics,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            target_variable_indices=[0],
        )
        assert model.multistage_only is False

    def test_processor_with_custom_loss_multistage_only(
        self,
        cfg_decoder: DictConfig,
        cfg_encoders: DictConfig,
        cfg_processor: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        cfg_processor = DictConfig(
            {**cfg_processor, "computes_loss_in_latent_space": True}
        )
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoders=cfg_encoders,
            processor=cfg_processor,
            decoder=cfg_decoder,
            hemisphere="north",
            input_spaces=[cfg_input_space],
            loss=cfg_loss,
            metrics=cfg_metrics,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            target_variable_indices=[0],
        )
        assert model.multistage_only is True


def test_latent_target_uses_target_input_encoder(
    cfg_decoder: DictConfig,
    cfg_encoders: DictConfig,
    cfg_input_space: DictConfig,
    cfg_loss: DictConfig,
    cfg_metrics: list[str],
) -> None:
    """Latent supervision must use the same target-dataset encoder as history."""
    output_space = DictConfig(
        {
            "channels": 1,
            "name": cfg_input_space["name"],
            "shape": cfg_input_space["shape"],
        }
    )
    processor = DictConfig(
        {
            "_target_": "icenet_mp.models.processors.NullProcessor",
            "computes_loss_in_latent_space": True,
        }
    )
    model = EncodeProcessDecode(
        name="shared-target-latent",
        encoders=cfg_encoders,
        processor=processor,
        decoder=cfg_decoder,
        hemisphere="north",
        input_spaces=[cfg_input_space],
        loss=cfg_loss,
        metrics=cfg_metrics,
        n_forecast_steps=2,
        n_history_steps=2,
        output_space=output_space,
        optimizer=DictConfig({}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
        target_variable_indices=[2],
    )
    model.eval()

    target_input_encoder = model.encoders[0]
    assert model.target_input_encoder is target_input_encoder
    assert model.processor.data_space_target == target_input_encoder.data_space_out
    assert model.processor.data_space_target != model.target_encoder.data_space_out

    history = torch.rand(2, 2, cfg_input_space["channels"], 16, 16)
    target = torch.rand(2, 2, 1, 16, 16)
    full_target = history[:, -1:].expand(-1, 2, -1, -1, -1).clone()
    full_target[:, :, 2:3] = target

    expected = target_input_encoder.rollout(full_target)
    actual = model.encode_target_latent({cfg_input_space["name"]: history}, target)

    torch.testing.assert_close(actual, expected)
