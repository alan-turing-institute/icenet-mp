import torch
from omegaconf import DictConfig

from icenet_mp.models.multistage import EncoderStage
from icenet_mp.types import DataSpace


class TestEncoderStage:
    def test_forward_ignores_skip_connection_persistence_requirement(
        self,
        cfg_encoders: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        cfg_loss: DictConfig,
    ) -> None:
        """A decoder configured with a skip connection must not require persistence.

        EncoderStage's disposable decoder reconstructs the encoder's own input (not the
        forecast target) and intentionally passes no persistence to `decoder.rollout`.
        A `model.decoder.skip_connection` override (e.g. `additive`) must not leak into
        this disposable decoder and blow up with a missing-persistence ValueError.
        """
        encoder_stage = EncoderStage(
            channel_names=["channel-0", "channel-1", "channel-2", "channel-3"],
            data_space_in=DataSpace.from_dict(cfg_input_space),
            encoder=cfg_encoders["test-input"],
            decoder=DictConfig(
                {
                    "_target_": "icenet_mp.models.decoders.NaiveLinearDecoder",
                    "skip_connection": {"method": "additive"},
                }
            ),
            latent_space=cfg_encoders["latent_space"],
            hemisphere="north",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            name="test-input_encoder",
            optimizer=cfg_optimizer,
            output_space=cfg_output_space,
            scheduler=cfg_scheduler,
            loss=cfg_loss,
        )

        assert encoder_stage.decoder.skip_connection is None

        batch_size = 2
        result = encoder_stage(
            {
                "target": torch.rand(
                    batch_size,
                    1,
                    cfg_input_space["channels"],
                    *cfg_input_space["shape"],
                )
            }
        )

        assert result.shape == (
            batch_size,
            1,
            cfg_input_space["channels"],
            *cfg_input_space["shape"],
        )
