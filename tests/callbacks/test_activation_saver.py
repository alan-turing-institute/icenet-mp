import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from lightning import LightningModule, Trainer
from torch import nn

from icenet_mp.callbacks import ActivationSaver


class ReusedLayerModel(nn.Module):
    """Tiny model that calls the same layer twice in one forward pass."""

    def __init__(self) -> None:
        """Initialise the repeated-layer test model."""
        super().__init__()
        self.shared = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.shared.weight.copy_(2 * torch.eye(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared layer twice."""
        return self.shared(self.shared(x))


class TestActivationSaver:
    def test_attach_rejects_unknown_layer(self, tmp_path: Path) -> None:
        """Reject unknown layer paths."""
        saver = ActivationSaver(["missing"], tmp_path)
        model = nn.Sequential(nn.Linear(2, 2))

        with pytest.raises(ValueError, match="missing"):
            saver.attach(model)

    def test_layer_hook_keeps_first_fire_per_batch(self, tmp_path: Path) -> None:
        """Keep only the first activation when a layer fires twice."""
        saver = ActivationSaver(["shared"], tmp_path)
        model = ReusedLayerModel()
        saver.attach(model)
        saver.on_test_batch_start(
            MagicMock(spec=Trainer), MagicMock(spec=LightningModule), {}, 0
        )

        inputs = torch.tensor([[1.0, 3.0]])
        output = model(inputs)

        assert torch.equal(output, 4 * inputs)
        assert torch.equal(saver._current_activations["shared"], 2 * inputs)
        saver.detach()

    def test_batch_payload_and_metadata_are_written(self, tmp_path: Path) -> None:
        """Write batch payloads and evaluation metadata."""
        saver = ActivationSaver(["0"], tmp_path, save_inputs=True)
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        saver.attach(model)

        trainer = MagicMock(spec=Trainer)
        pl_module = MagicMock(spec=LightningModule)
        sic_batch = torch.tensor([[1.0, 2.0]])
        batch = {
            "sic": sic_batch,
            "metadata": "not-a-tensor",
        }
        saver.on_test_batch_start(trainer, pl_module, batch, 7)
        model(sic_batch)
        saver.on_test_batch_end(trainer, pl_module, None, batch, 7)

        payload_path = tmp_path / "batch_00007.pt"
        payload = torch.load(payload_path, weights_only=False)

        assert payload["batch_idx"] == 7
        assert payload["layer_paths"] == ["0"]
        assert set(payload["activations"]) == {"0"}
        assert set(payload["inputs"]) == {"sic"}
        assert torch.equal(payload["inputs"]["sic"], sic_batch)

        saver.on_test_end(trainer, pl_module)
        metadata = json.loads((tmp_path / "metadata.json").read_text())

        assert metadata["layer_paths"] == ["0"]
        assert metadata["save_inputs"] is True
        assert metadata["batch_file_template"] == "batch_{batch_idx:05d}.pt"
        assert saver._handles == []

    def test_empty_layer_list_is_disabled(self, tmp_path: Path) -> None:
        """Leave the callback disabled when no layers are configured."""
        output_dir = tmp_path / "activations"
        saver = ActivationSaver([], output_dir)

        trainer = MagicMock(spec=Trainer)
        pl_module = MagicMock(spec=LightningModule)
        saver.on_test_start(trainer, pl_module)
        saver.on_test_batch_start(trainer, pl_module, {}, 0)
        saver.on_test_batch_end(trainer, pl_module, None, {}, 0)
        saver.on_test_end(trainer, pl_module)

        assert not output_dir.exists()
