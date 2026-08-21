import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.models import EncodeProcessDecode
from icenet_mp.types import ProcessorOutput


def test_uncertainty_loss_rejects_processor_owned_training_loss(
    cfg_decoder: DictConfig,
    cfg_encoders: DictConfig,
    cfg_processor: DictConfig,
    cfg_input_space: DictConfig,
    cfg_output_space: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject uncertainty weighting when the processor supplies its own loss."""
    model = EncodeProcessDecode(
        name="encode-null-decode",
        encoders=cfg_encoders,
        processor=cfg_processor,
        decoder=cfg_decoder,
        hemisphere="north",
        input_spaces=[cfg_input_space],
        loss=DictConfig(
            {"_target_": "icenet_mp.losses.UncertaintyWeightedLoss", "delta": 0.5}
        ),
        n_forecast_steps=1,
        n_history_steps=1,
        output_space=cfg_output_space,
        optimizer=DictConfig({}),
        scheduler=DictConfig({}),
        target_variable_indices=[0],
    )
    processor_output = ProcessorOutput(
        prediction=torch.empty(0),
        loss=torch.tensor(1.0),
    )
    monkeypatch.setattr(
        model.processor,
        "rollout",
        lambda *args, **kwargs: processor_output,  # noqa: ARG005
    )
    target = torch.rand(
        1,
        1,
        cfg_output_space["channels"],
        cfg_output_space["shape"][0],
        cfg_output_space["shape"][1],
    )
    batch = {
        cfg_input_space["name"]: torch.randn(
            1,
            1,
            cfg_input_space["channels"],
            cfg_input_space["shape"][0],
            cfg_input_space["shape"][1],
        ),
        "target": target,
        "target_uncertainty": torch.full_like(target, 0.1),
    }

    with pytest.raises(ValueError, match="owns the training loss in latent space"):
        model.training_step(batch, 0)


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
            loss=cfg_loss,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
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

    def test_processor_default_does_not_require_multistage(  # noqa: PLR0917
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
            loss=cfg_loss,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            target_variable_indices=[0],
        )
        assert model.multistage_only is False

    def test_processor_with_custom_loss_multistage_only(  # noqa: PLR0917
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
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            target_variable_indices=[0],
        )
        assert model.multistage_only is True
