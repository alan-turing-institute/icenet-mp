from unittest.mock import MagicMock

import pytest
from lightning import LightningModule, Trainer


@pytest.fixture
def mock_trainer() -> MagicMock:
    """A bare mock Trainer; tests configure whatever attributes they need."""
    return MagicMock(spec=Trainer)


@pytest.fixture
def mock_module() -> MagicMock:
    """A bare mock LightningModule; tests configure whatever attributes they need."""
    return MagicMock(spec=LightningModule)
