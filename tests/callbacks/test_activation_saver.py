"""Tests for evaluation activation persistence."""

from pathlib import Path
from types import SimpleNamespace

import torch

from icenet_mp.callbacks.activation_saver import ActivationSaver


def test_distributed_batches_have_unique_paths(tmp_path: Path) -> None:
    """Evaluation ranks must not overwrite one another's activation batches."""
    for rank in (0, 1):
        saver = ActivationSaver(["processor"], tmp_path, save_inputs=False)
        saver._current_activations = {"processor": torch.tensor([rank])}
        trainer = SimpleNamespace(global_rank=rank, world_size=2)

        saver.on_test_batch_end(
            trainer,  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
            None,
            None,
            batch_idx=0,
        )

    paths = sorted(tmp_path.glob("*.pt"))
    assert [path.name for path in paths] == [
        "rank_000_dataloader_000_batch_00000.pt",
        "rank_001_dataloader_000_batch_00000.pt",
    ]
    assert [torch.load(path)["global_rank"] for path in paths] == [0, 1]


def test_only_global_rank_writes_metadata(tmp_path: Path) -> None:
    """A non-zero evaluation rank must not race when writing shared metadata."""
    saver = ActivationSaver(["processor"], tmp_path, save_inputs=False)
    trainer = SimpleNamespace(is_global_zero=False)

    saver.on_test_end(
        trainer,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
    )

    assert not (tmp_path / saver.METADATA_FILE).exists()
