from collections.abc import Iterator

import pytest
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(autouse=True)
def _clear_global_hydra() -> Iterator[None]:
    """Reset Hydra's global singleton state before and after each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()
