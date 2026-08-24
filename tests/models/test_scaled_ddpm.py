import pytest
import torch

from icenet_mp.models.processors import DDPMProcessor, ScaledDDPMProcessor
from icenet_mp.types import DataSpace, ProcessorOutput


def _make_processor(*, x0_scale: float = 2.0) -> ScaledDDPMProcessor:
    combined = DataSpace(name="combined", channels=4, shape=(8, 8))
    target = DataSpace(name="target", channels=2, shape=(8, 8))
    return ScaledDDPMProcessor(
        data_space=combined,
        data_space_target=target,
        n_forecast_steps=1,
        n_history_steps=1,
        timesteps=2,
        start_out_channels=8,
        time_embed_dim=256,
        dropout_rate=0.0,
        use_autoregressive=True,
        target_channel_offset=1,
        loss=torch.nn.MSELoss(),
        x0_scale=x0_scale,
    )


def test_scaled_ddpm_rejects_non_positive_x0_scale() -> None:
    with pytest.raises(ValueError, match="x0_scale must be positive"):
        ScaledDDPMProcessor(x0_scale=0.0)


def test_scaled_ddpm_scales_training_target_and_restores_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor(x0_scale=2.0)
    x = torch.randn(1, 1, 4, 8, 8)
    y = torch.full((1, 1, 2, 8, 8), 4.0)
    captured: dict[str, torch.Tensor] = {}

    def fake_rollout_training(
        self: DDPMProcessor, x_input: torch.Tensor, y_input: torch.Tensor
    ) -> ProcessorOutput:
        captured["target"] = y_input.clone()
        prediction = torch.zeros(1, 1, 4, 8, 8)
        prediction[..., self.target_slice_start : self.target_slice_end, :, :] = 3.0
        return ProcessorOutput(prediction=prediction, loss=torch.tensor(1.0))

    monkeypatch.setattr(DDPMProcessor, "_rollout_training", fake_rollout_training)

    output = processor._rollout_training(x, y)

    torch.testing.assert_close(captured["target"], y / 2.0)
    torch.testing.assert_close(
        output.prediction[..., processor.target_slice_start : processor.target_slice_end, :, :],
        torch.full((1, 1, 2, 8, 8), 6.0),
    )
    torch.testing.assert_close(output.prediction[..., 0, :, :], torch.zeros(1, 1, 8, 8))


def test_scaled_ddpm_restores_scale_after_reverse_diffusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor(x0_scale=2.5)
    noisy = torch.randn(1, 2, 8, 8)
    cond = torch.randn(1, 4, 8, 8)

    def fake_reverse_diffusion(
        self: DDPMProcessor, y: torch.Tensor, cond_input: torch.Tensor
    ) -> torch.Tensor:
        del self, cond_input
        return torch.ones_like(y)

    monkeypatch.setattr(DDPMProcessor, "_run_reverse_diffusion", fake_reverse_diffusion)

    output = processor._run_reverse_diffusion(noisy, cond)

    torch.testing.assert_close(output, torch.full_like(noisy, 2.5))
