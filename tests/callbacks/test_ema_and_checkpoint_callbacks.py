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


def test_unconditional_checkpoint_dirpath_roundtrip(tmp_path: Path) -> None:
    callback = UnconditionalCheckpoint()

    callback.dirpath = tmp_path

    assert callback.dirpath == tmp_path


def test_unconditional_checkpoint_uses_log_dir_and_broadcast(tmp_path: Path) -> None:
    callback = UnconditionalCheckpoint()
    callback.impl.format_checkpoint_name = MagicMock(return_value="epoch=2-step=7.ckpt")
    trainer = MagicMock()
    trainer.current_epoch = 2
    trainer.global_step = 7
    trainer.log_dir = str(tmp_path)
    trainer.strategy.broadcast.side_effect = lambda value: value

    callback.save_unconditionally(trainer)

    callback.impl.format_checkpoint_name.assert_called_once()
    expected = tmp_path / "epoch=2-step=7.ckpt"
    trainer.strategy.broadcast.assert_called_once_with(str(expected))
    trainer.save_checkpoint.assert_called_once_with(expected)


def test_unconditional_checkpoint_prefers_explicit_dirpath(tmp_path: Path) -> None:
    callback = UnconditionalCheckpoint()
    explicit = tmp_path / "checkpoints"
    callback.dirpath = explicit
    callback.impl.format_checkpoint_name = MagicMock(return_value="last.ckpt")
    trainer = MagicMock()
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.log_dir = str(tmp_path / "logs")
    trainer.strategy.broadcast.side_effect = lambda value: value

    callback.save_unconditionally(trainer)

    trainer.save_checkpoint.assert_called_once_with(explicit / "last.ckpt")


def test_unconditional_checkpoint_train_end_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = MagicMock()
    module = MagicMock()

    disabled = UnconditionalCheckpoint(on_train_end=False)
    disabled_save = MagicMock()
    monkeypatch.setattr(disabled, "save_unconditionally", disabled_save)
    disabled.on_train_end(trainer, module)
    disabled_save.assert_not_called()

    enabled = UnconditionalCheckpoint(on_train_end=True)
    enabled_save = MagicMock()
    monkeypatch.setattr(enabled, "save_unconditionally", enabled_save)
    enabled.on_train_end(trainer, module)
    enabled_save.assert_called_once_with(trainer)
