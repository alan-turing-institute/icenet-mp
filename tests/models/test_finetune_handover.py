from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning import Trainer
from omegaconf import DictConfig

from icenet_mp.model_service import ModelService
from icenet_mp.models import EncodeProcessDecode
from icenet_mp.models.multistage import DecoderStage, EncoderStage, ProcessorStage


@pytest.fixture
def handover_models(tmp_path: Path) -> Iterator[tuple[ModelService, ProcessorStage]]:
    """Build real stages with distinguishable target weights and BatchNorm buffers."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        space = DictConfig({"name": "sic", "channels": 1, "shape": [16, 16]})
        encoder = DictConfig(
            {
                "_target_": "icenet_mp.models.encoders.CNNEncoder",
                "n_layers": 1,
                "n_subblocks": 1,
                "activation": "LeakyReLU",
            }
        )
        decoder = DictConfig(
            {"_target_": "icenet_mp.models.decoders.NaiveLinearDecoder"}
        )
        processor = DictConfig(
            {
                "_target_": "icenet_mp.models.processors.DDPMProcessor",
                "timesteps": 2,
                "start_out_channels": 8,
                "time_embed_dim": 256,
                "normalization": "none",
                "dropout_rate": 0.0,
                "loss": {"_target_": "torch.nn.MSELoss"},
            }
        )
        model = EncodeProcessDecode(
            encoders=DictConfig({"latent_space": [16, 16], "sic": encoder}),
            processor=processor,
            decoder=decoder,
            target_variable_indices=[0],
            hemisphere="north",
            input_spaces=[space],
            output_space=space,
            mask_dir=str(tmp_path),
            n_history_steps=2,
            n_forecast_steps=2,
            name="finetune-handover",
            metrics=[],
            loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
            optimizer=DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-3}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
        )
        input_stage = EncoderStage.from_template(
            channel_names=["sic"],
            data_space_in=model.input_spaces[0],
            dataset="sic",
            decoder=decoder,
            encoder=encoder,
            template=model,
        )
        target_stage = EncoderStage.from_template(
            channel_names=["sic"],
            data_space_in=model.target_encoder.data_space_in,
            dataset="target",
            decoder=decoder,
            encoder=encoder,
            template=model,
        )
        decoder_stage = DecoderStage.from_template(
            decoder=decoder,
            encoders=[input_stage],
            target_dataset_name="sic",
            target_variable_indices=[0],
        )
        pretrained = ProcessorStage.from_template(
            processor=processor,
            decoder_model=decoder_stage,
            target_encoder=target_stage,
        )
        # Stand in for checkpoint contents without running an optimisation step.
        with torch.no_grad():
            for parameter in pretrained.target_encoder.parameters():
                parameter.add_(0.5)
            for buffer in pretrained.target_encoder.buffers():
                buffer.add_(2)
        service = ModelService.__new__(ModelService)
        service.model_ = model
        yield service, pretrained


def test_finetune_transfers_parameters_and_buffers_before_fitting(
    handover_models: tuple[ModelService, ProcessorStage],
) -> None:
    """Copy the full state without replacing the encoder with a frozen module."""
    service, pretrained = handover_models
    model = service.model
    assert isinstance(model, EncodeProcessDecode)
    target_encoder = model.target_encoder
    trainable = [p.requires_grad for p in target_encoder.parameters()]
    training_mode = target_encoder.training
    assert trainable
    assert all(trainable)
    buffer_names = dict(pretrained.target_encoder.named_buffers())
    assert any(name.endswith("running_mean") for name in buffer_names)
    assert any(name.endswith("running_var") for name in buffer_names)
    assert any(name.endswith("num_batches_tracked") for name in buffer_names)
    trainer = MagicMock(spec=Trainer)
    config = DictConfig({})

    def check_handover(*, config: DictConfig, job_stage: str) -> Trainer:
        assert job_stage == "finetune"
        assert config == DictConfig({})
        for actual, expected in zip(
            (*model.encoders, model.processor, model.decoder, model.target_encoder),
            (
                *pretrained.encoders,
                pretrained.processor,
                pretrained.decoder,
                pretrained.target_encoder,
            ),
            strict=True,
        ):
            torch.testing.assert_close(
                actual.state_dict(), expected.state_dict(), rtol=0, atol=0
            )
        assert model.target_encoder is target_encoder
        assert [p.requires_grad for p in target_encoder.parameters()] == trainable
        assert target_encoder.training == training_mode
        assert not any(p.requires_grad for p in pretrained.target_encoder.parameters())
        for actual_tensor, expected_tensor in zip(
            (*target_encoder.parameters(), *target_encoder.buffers()),
            (
                *pretrained.target_encoder.parameters(),
                *pretrained.target_encoder.buffers(),
            ),
            strict=True,
        ):
            assert actual_tensor.data_ptr() != expected_tensor.data_ptr()
        return trainer

    with (
        patch.object(service, "_fit", side_effect=check_handover) as fit,
        patch.object(service, "_save_stage_checkpoint") as save,
    ):
        result = service.train_stage_finetune(config=config, processor_model=pretrained)
    fit.assert_called_once_with(config=config, job_stage="finetune")
    save.assert_called_once_with(trainer, "finetune")
    assert result is trainer


@pytest.mark.parametrize("seed", [0, 99, 123])
def test_finetune_preserves_ddpm_objective_before_optimisation(
    handover_models: tuple[ModelService, ProcessorStage], seed: int
) -> None:
    """The same diffusion randomness must give the same loss across the handover."""
    service, pretrained = handover_models
    model = service.model
    assert isinstance(model, EncodeProcessDecode)
    model.eval()
    pretrained.eval()
    inputs = torch.rand(2, 2, 1, 16, 16)
    targets = torch.rand(2, 2, 1, 16, 16)

    def loss(module: EncodeProcessDecode) -> torch.Tensor:
        with (
            torch.random.fork_rng(devices=[]),
            torch.no_grad(),
            patch.object(module, "log"),
        ):
            torch.manual_seed(seed)
            return module.training_step({"sic": inputs, "target": targets}, 0).loss

    expected = loss(pretrained)
    with patch.object(service, "_fit"), patch.object(service, "_save_stage_checkpoint"):
        service.train_stage_finetune(config=DictConfig({}), processor_model=pretrained)
    assert torch.isfinite(expected)
    torch.testing.assert_close(loss(model), expected, rtol=0, atol=0)
