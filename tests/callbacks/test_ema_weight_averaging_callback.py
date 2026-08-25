from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from lightning.pytorch.callbacks import WeightAveraging

from icenet_mp.callbacks.ema_weight_averaging_callback import EMAWeightAveragingCallback
from icenet_mp.callbacks.unconditional_checkpoint import UnconditionalCheckpoint


@pytest.mark.parametrize(
    ("step_idx", "epoch_idx", "expected"),
    [
        (2, None, False),
        (3, None, True),
        (6, None, True),
        (None, 1, False),
        (None, 2, True),
        (None, 4, True),
        (None, None, False),
    ],
)
def test_ema_should_update_on_configured_interval(
    step_idx: int | None, epoch_idx: int | None, *, expected: bool
) -> None:
    callback = EMAWeightAveragingCallback(
        decay_rate=0.99,
        every_n_steps=3,
        every_n_epochs=2,
    )

    assert callback.should_update(step_idx=step_idx, epoch_idx=epoch_idx) is expected


def test_ema_batch_hook_skips_parameterless_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = EMAWeightAveragingCallback(decay_rate=0.99, every_n_steps=1)
    parent_hook = MagicMock()
    monkeypatch.setattr(WeightAveraging, "on_train_batch_end", parent_hook)
    module = torch.nn.Identity()

    callback.on_train_batch_end(MagicMock(), module)

    parent_hook.assert_not_called()


def test_ema_hooks_delegate_for_parameterised_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = EMAWeightAveragingCallback(decay_rate=0.99, every_n_steps=1)
    batch_hook = MagicMock()
    epoch_hook = MagicMock()
    monkeypatch.setattr(WeightAveraging, "on_train_batch_end", batch_hook)
    monkeypatch.setattr(WeightAveraging, "on_train_epoch_end", epoch_hook)
    module = torch.nn.Linear(2, 2)
    trainer = MagicMock()

    callback.on_train_batch_end(trainer, module)
    callback.on_train_epoch_end(trainer, module)

    batch_hook.assert_called_once()
    epoch_hook.assert_called_once()
