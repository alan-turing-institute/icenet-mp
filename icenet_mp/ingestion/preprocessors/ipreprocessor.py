from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any


class IPreprocessor(ABC):
    def __init__(self, config: MutableMapping[str, Any]) -> None:
        """Initialise the IPreprocessor base class."""
        self.cls_name = str(config.get("preprocessor", {}).get("type", "None"))
        self.dataset_name = str(config.get("name", "None"))

    @abstractmethod
    def download(self, preprocessor_path: Path) -> None:
        """Download data to the specified preprocessor path."""
