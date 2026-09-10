from unittest.mock import MagicMock

import pytest
import torch
from lightning import LightningModule, Trainer


class LinearLightningModule(LightningModule):
    """A LightningModule wrapping a single trainable Linear(2, 2) layer."""

    def __init__(self) -> None:
        """Initialise a LightningModule with one trainable layer."""
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)


@pytest.fixture
def linear_lightning_module() -> LinearLightningModule:
    """A real LightningModule with one trainable layer."""
    return LinearLightningModule()


@pytest.fixture
def mock_module() -> MagicMock:
    """A bare mock LightningModule; tests configure whatever attributes they need."""
    return MagicMock(spec=LightningModule)


@pytest.fixture
def mock_trainer() -> MagicMock:
    """A bare mock Trainer; tests configure whatever attributes they need."""
    return MagicMock(spec=Trainer)
