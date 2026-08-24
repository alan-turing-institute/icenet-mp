import json
from pathlib import Path

import pytest
import torch
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


def test_attach_rejects_unknown_layer(tmp_path: Path) -> None:
    """Reject unknown layer paths."""
    saver = ActivationSaver(["missing"], tmp_path)
    model = nn.Sequential(nn.Linear(2, 2))

    with pytest.raises(ValueError, match="missing"):
        saver.attach(model)


def test_layer_hook_keeps_first_fire_per_batch(tmp_path: Path) -> None:
    """Keep only the first activation when a layer fires twice."""
    saver = ActivationSaver(["shared"], tmp_path)
    model = ReusedLayerModel()
    saver.attach(model)
    saver.on_test_batch_start(None, None, {}, 0)  # type: ignore[arg-type]

    inputs = torch.tensor([[1.0, 3.0]])
    output = model(inputs)

    torch.testing.assert_close(output, 4 * inputs)
    torch.testing.assert_close(saver._current_activations["shared"], 2 * inputs)
    saver.detach()


def test_batch_payload_and_metadata_are_written(tmp_path: Path) -> None:
    """Write batch payloads and evaluation metadata."""
    saver = ActivationSaver(["0"], tmp_path, save_inputs=True)
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    saver.attach(model)

    batch = {
        "sic": torch.tensor([[1.0, 2.0]]),
        "metadata": "not-a-tensor",
    }
    saver.on_test_batch_start(None, None, batch, 7)  # type: ignore[arg-type]
    model(batch["sic"])
    saver.on_test_batch_end(None, None, None, batch, 7)  # type: ignore[arg-type]

    payload_path = tmp_path / "batch_00007.pt"
    payload = torch.load(payload_path, weights_only=False)

    assert payload["batch_idx"] == 7
    assert payload["layer_paths"] == ["0"]
    assert set(payload["activations"]) == {"0"}
    assert set(payload["inputs"]) == {"sic"}
    torch.testing.assert_close(payload["inputs"]["sic"], batch["sic"])

    saver.on_test_end(None, None)  # type: ignore[arg-type]
    metadata = json.loads((tmp_path / "metadata.json").read_text())

    assert metadata["layer_paths"] == ["0"]
    assert metadata["save_inputs"] is True
    assert metadata["batch_file_template"] == "batch_{batch_idx:05d}.pt"
    assert saver._handles == []


def test_empty_layer_list_is_disabled(tmp_path: Path) -> None:
    """Leave the callback disabled when no layers are configured."""
    output_dir = tmp_path / "activations"
    saver = ActivationSaver([], output_dir)

    saver.on_test_start(None, None)  # type: ignore[arg-type]
    saver.on_test_batch_start(None, None, {}, 0)  # type: ignore[arg-type]
    saver.on_test_batch_end(None, None, None, {}, 0)  # type: ignore[arg-type]
    saver.on_test_end(None, None)  # type: ignore[arg-type]

    assert not output_dir.exists()
